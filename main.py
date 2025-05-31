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

# --- Configuration ---
SAMPLE_RATE = 16000  # AST typically uses 16kHz
BLOCK_SIZE = 1024    
WINDOW_LENGTH_SECONDS = 1.0  # Need longer window for meaningful classification
DISPLAY_WINDOW_SECONDS = 2   
PREDICTION_INTERVAL = 1.0  # Predict every 1 second

# Calculate derived parameters
WINDOW_LENGTH_SAMPLES = int(SAMPLE_RATE * WINDOW_LENGTH_SECONDS)
DISPLAY_WINDOW_SAMPLES = int(SAMPLE_RATE * DISPLAY_WINDOW_SECONDS)

# Data buffers
audio_buffer = deque(maxlen=DISPLAY_WINDOW_SAMPLES)  # For waveform display
prediction_buffer = deque(maxlen=WINDOW_LENGTH_SAMPLES)  # For classification

# Lock for thread-safe access
buffer_lock = threading.Lock()

# Flags and state
recording_active = threading.Event()
recording_active.set()
current_prediction = "Listening..."
last_prediction_time = 0

# --- Load AST Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
model.eval()

# --- Audio Processing Functions ---
def preprocess_audio(audio_array):
    """Convert audio array to model input format"""
    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    return inputs.to(device)

def predict_audio(audio_array):
    """Run inference on audio chunk"""
    with torch.no_grad():
        inputs = preprocess_audio(audio_array)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
        return model.config.id2label[pred_idx], confidence

# --- Audio Callback Function ---
def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each incoming audio block"""
    global current_prediction, last_prediction_time
    
    if status:
        print(status)
    if not recording_active.is_set():
        raise sd.CallbackStop 
    
    current_block = indata[:, 0] if indata.ndim > 1 else indata
    
    with buffer_lock:
        # Always update display buffer for smooth waveform
        audio_buffer.extend(current_block)
        
        # Update prediction buffer
        prediction_buffer.extend(current_block)
        
        # Run prediction periodically if we have enough audio
        current_time = time.time()
        if (current_time - last_prediction_time) > PREDICTION_INTERVAL and len(prediction_buffer) >= WINDOW_LENGTH_SAMPLES:
            try:
                processing_window = np.array(prediction_buffer)
                label, confidence = predict_audio(processing_window)
                current_prediction = f"{label} ({confidence:.2f})"
                last_prediction_time = current_time
                # Clear prediction buffer after successful prediction
                prediction_buffer.clear()
            except Exception as e:
                print(f"Prediction error: {e}")
                current_prediction = f"Prediction error: {e}"

# --- Plotting Setup ---
fig, (ax_wave, ax_text) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
line, = ax_wave.plot(np.zeros(DISPLAY_WINDOW_SAMPLES), color='blue')
ax_wave.set_ylim([-1, 1])
ax_wave.set_xlim([0, DISPLAY_WINDOW_SAMPLES])
ax_wave.set_title("Real-time Audio Waveform")
ax_wave.set_xlabel("Samples")
ax_wave.set_ylabel("Amplitude")

prediction_text = ax_text.text(0.5, 0.5, current_prediction, 
                             ha='center', va='center', fontsize=12)
ax_text.axis('off')

# --- Update function for FuncAnimation ---
def update_plot(frame):
    with buffer_lock: 
        # Always update waveform display from audio_buffer
        if len(audio_buffer) > 0:
            plot_data = np.array(audio_buffer)
            if len(plot_data) < DISPLAY_WINDOW_SAMPLES:
                # Pad with zeros if buffer not full
                padded_data = np.zeros(DISPLAY_WINDOW_SAMPLES)
                padded_data[:len(plot_data)] = plot_data
                line.set_ydata(padded_data)
            else:
                line.set_ydata(plot_data)
        
        # Update prediction text
        prediction_text.set_text(current_prediction)
    
    return line, prediction_text

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Recording audio at {SAMPLE_RATE} Hz with a block size of {BLOCK_SIZE} frames.")
    print(f"Displaying {DISPLAY_WINDOW_SECONDS} seconds of audio.")
    print("Press Ctrl+C to stop recording.")

    try:
        # Start the audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            callback=audio_callback
        )

        # Animation updates every 50ms for smooth display
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