import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import joblib
import librosa
from scipy.fft import fft

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
WINDOW_LENGTH_SECONDS = 1.0
DISPLAY_WINDOW_SECONDS = 2
PREDICTION_INTERVAL = 1.0

WINDOW_LENGTH_SAMPLES = int(SAMPLE_RATE * WINDOW_LENGTH_SECONDS)
DISPLAY_WINDOW_SAMPLES = int(SAMPLE_RATE * DISPLAY_WINDOW_SECONDS)

audio_buffer = deque(maxlen=DISPLAY_WINDOW_SAMPLES)
prediction_buffer = deque(maxlen=WINDOW_LENGTH_SAMPLES)

buffer_lock = threading.Lock()

recording_active = threading.Event()
recording_active.set()
current_ast_prediction = "Listening..."
current_rf_prediction = "Waiting for RF..."
last_prediction_time = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
model.eval()

RF_MODEL_FILENAME = 'random_forest_best_model.joblib'
rf_model = None
rf_label_mapping = {0: 'laugh', 1: 'cough', 2: 'clap', 3: 'knock', 4: 'alarm'}

try:
    rf_model = joblib.load(RF_MODEL_FILENAME)
    print(f"Random Forest model '{RF_MODEL_FILENAME}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: Random Forest model file '{RF_MODEL_FILENAME}' not found. RF predictions will be unavailable.")
except Exception as e:
    print(f"An error occurred while loading the Random Forest model: {e}. RF predictions will be unavailable.")

def extract_fft(audio_signal, sr=16000):
    n_fft_features=4

    if audio_signal.size == 0:
        return np.zeros(n_fft_features)

    fft_result = np.fft.rfft(audio_signal)
    magnitude_spectrum = np.abs(fft_result)

    if magnitude_spectrum.size == 0:
         return np.zeros(n_fft_features)

    fft_mean = np.mean(magnitude_spectrum)
    fft_std = np.std(magnitude_spectrum)
    fft_max = np.max(magnitude_spectrum)
    fft_min = np.min(magnitude_spectrum)

    feature_vector = np.array([fft_mean, fft_std, fft_max, fft_min])

    if len(feature_vector) < n_fft_features:
       feature_vector = np.pad(feature_vector, (0, n_fft_features - len(feature_vector)))
    elif len(feature_vector) > n_fft_features:
        feature_vector = feature_vector[:n_fft_features]

    return feature_vector

def extract_mfcc(audio_signal, sr=16000):
    n_mfcc = 13
    include_deltas = True

    features = []
    expected_len = n_mfcc * 2
    if include_deltas:
        expected_len *= 3

    try:
        audio_signal = audio_signal.astype(np.float32)

        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features.extend([mfccs_mean, mfccs_std])

        if include_deltas:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)

            delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
            delta_mfccs_std = np.std(delta_mfccs, axis=1)
            features.extend([delta_mfccs_mean, delta_mfccs_std])

            delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)
            delta2_mfccs_std = np.std(delta2_mfccs, axis=1)
            features.extend([delta2_mfccs_mean, delta2_mfccs_std])

        feature_vector = np.hstack(features)

        if len(feature_vector) != expected_len:
             if len(feature_vector) < expected_len:
                 feature_vector = np.pad(feature_vector, (0, expected_len - len(feature_vector)))
             else:
                 feature_vector = feature_vector[:expected_len]

        return feature_vector

    except Exception as e:
        return np.zeros(expected_len)

def extract_rms(audio_signal, sr=16000):
    n_rms_features = 4
    hop_length = 512
    frame_length = 2048

    try:
        audio_signal = audio_signal.astype(np.float32)

        rms = librosa.feature.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]

        if rms.size == 0:
            return np.zeros(n_rms_features)

        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms)
        rms_min = np.min(rms)

        feature_vector = np.array([rms_mean, rms_std, rms_max, rms_min])

        if len(feature_vector) < n_rms_features:
           feature_vector = np.pad(feature_vector, (0, n_rms_features - len(feature_vector)))
        elif len(feature_vector) > n_rms_features:
           feature_vector = feature_vector[:n_rms_features]

        return feature_vector

    except Exception as e:
        return np.zeros(n_rms_features)

def extract_features_for_rf_realtime(audio_segment, sr):
    fft_feats = extract_fft(audio_segment, sr)
    mfcc_feats = extract_mfcc(audio_segment, sr)
    rms_feats = extract_rms(audio_segment, sr)

    combined_features = np.concatenate([
        fft_feats,
        mfcc_feats,
        rms_feats
    ])

    return combined_features.reshape(1, -1)

def preprocess_audio_ast(audio_array):
    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    return inputs.to(device)

def predict_audio_ast(audio_array):
    with torch.no_grad():
        inputs = preprocess_audio_ast(audio_array)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
        return model.config.id2label[pred_idx], confidence

def audio_callback(indata, frames, time_info, status):
    global current_ast_prediction, current_rf_prediction, last_prediction_time, rf_label_mapping

    if status:
        print(status)
    if not recording_active.is_set():
        raise sd.CallbackStop

    current_block = indata[:, 0] if indata.ndim > 1 else indata

    with buffer_lock:
        audio_buffer.extend(current_block)
        prediction_buffer.extend(current_block)

        current_time = time.time()
        if (current_time - last_prediction_time) > PREDICTION_INTERVAL and len(prediction_buffer) >= WINDOW_LENGTH_SAMPLES:
            try:
                processing_window = np.array(prediction_buffer)

                ast_label, ast_confidence = predict_audio_ast(processing_window)
                current_ast_prediction = f"{ast_label} ({ast_confidence:.2f})"

                if rf_model is not None:
                    try:
                        rf_features = extract_features_for_rf_realtime(processing_window, SAMPLE_RATE)

                        rf_prediction_label_idx = rf_model.predict(rf_features)[0]
                        rf_prediction_mapped = rf_label_mapping.get(rf_prediction_label_idx, f"Unknown ({rf_prediction_label_idx})")

                        if hasattr(rf_model, 'predict_proba'):
                            rf_probs = rf_model.predict_proba(rf_features)[0]
                            rf_confidence = np.max(rf_probs)
                            current_rf_prediction = f"{rf_prediction_mapped} ({rf_confidence:.2f})"
                        else:
                            current_rf_prediction = f"{rf_prediction_mapped} (No Prob.)"

                    except Exception as e:
                        current_rf_prediction = f"RF Error: {e}"
                else:
                    current_rf_prediction = "RF model not loaded."

                last_prediction_time = current_time
                prediction_buffer.clear()
            except Exception as e:
                print(f"Overall Prediction error: {e}")
                current_ast_prediction = f"AST Error: {e}"
                current_rf_prediction = "RF Error: No Prediction"

fig, (ax_wave, ax_text) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
line, = ax_wave.plot(np.zeros(DISPLAY_WINDOW_SAMPLES), color='blue')
ax_wave.set_ylim([-1, 1])
ax_wave.set_xlim([0, DISPLAY_WINDOW_SAMPLES])
ax_wave.set_title("Real-time Audio Waveform")
ax_wave.set_xlabel("Samples")
ax_wave.set_ylabel("Amplitude")

prediction_text = ax_text.text(0.5, 0.5, f"AST: {current_ast_prediction}\n{current_rf_prediction}",
                             ha='center', va='center', fontsize=12)
ax_text.axis('off')

def update_plot(frame):
    with buffer_lock:
        if len(audio_buffer) > 0:
            plot_data = np.array(audio_buffer)
            if len(plot_data) < DISPLAY_WINDOW_SAMPLES:
                padded_data = np.zeros(DISPLAY_WINDOW_SAMPLES)
                padded_data[:len(plot_data)] = plot_data
                line.set_ydata(padded_data)
            else:
                line.set_ydata(plot_data)

        prediction_text.set_text(f"AST: {current_ast_prediction}\nRF: {current_rf_prediction}")

    return line, prediction_text

if __name__ == "__main__":
    print(f"Recording audio at {SAMPLE_RATE} Hz with a block size of {BLOCK_SIZE} frames.")
    print(f"Displaying {DISPLAY_WINDOW_SECONDS} seconds of audio.")
    print("Press Ctrl+C to stop recording.")

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            callback=audio_callback
        )

        ani = animation.FuncAnimation(
            fig, update_plot, interval=50, blit=False, cache_frame_data=False
        )

        with stream:
            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("\nStopping recording...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        recording_active.clear()
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        plt.close(fig)
        print("Recording stopped and resources released.")