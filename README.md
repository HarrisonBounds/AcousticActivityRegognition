# AcousticActivityRegognition

### How to Run

1. Clone the repo

```
git clone git@github.com:HarrisonBounds/AcousticActivityRegognition.git
```

2. Create a virtual environment

```
cd AcousticActivityRegognition
python3 -m venv audio_recognition_env
```

3. Activate environment and install the  `requirements.txt`
```
source audio_recognition_env/bin/activate
pip install -r requirements.txt
```

3. Run the `main.py` script

```
python main.py
```
---

### Pipeline

This system operates as a real-time audio classification pipeline, continuously monitoring incoming sound and providing immediate insights.

It begins with audio capture, where the `sounddevice` library interfaces with the computer's microphone to acquire raw audio signals in small, fixed-size blocks. These blocks are then efficiently managed by a deque  buffer, ensuring that a consistent "window" of the most recent audio data is always available for subsequent processing. This raw audio stream forms the foundation for all downstream analytical steps.

Following capture, the pipeline proceeds to processing and inference. The buffered audio segment is fed into two distinct machine learning models: a pre-trained AST (Audio Spectrogram Transformer) model and a custom Random Forest model from a previous assignment. For the Random Forest, dedicated feature extraction functions (FFT, MFCC, RMS) transform the raw audio into a comprehensive numerical feature vector, which is then passed to the RF model for prediction. Concurrently, the AST model directly processes the audio (after its own internal feature extraction via feature_extractor) to classify the sound. Both models output a prediction label, confidence score, and their processing latencies.

Finally, the results are presented in a display module. A matplotlib figure, divided into two subplots, visually represents the live audio waveform and the real-time classification results. The waveform plot dynamically updates to show the amplitude of the incoming sound, providing a visual representation of the audio being processed. Below it, a text display presents the predictions from both the AST and Random Forest models, along with their respective confidence levels and the measured processing latencies in milliseconds. This continuous feedback loop allows users to observe the audio, the model's classifications, and the system's responsiveness.

---

### Buffering and Overlapping Windows

Incoming audio is captured in small, constant `BLOCK_SIZE` chunks by the `sounddevice` callback, which are then appended to two deque buffers: audio_buffer and prediction_buffer. The audio_buffer maintains a rolling window of the `DISPLAY_WINDOW_SAMPLES` (2 seconds) for smooth visual waveform display. Crucially, the prediction_buffer accumulates audio until it reaches `WINDOW_LENGTH_SAMPLES` (1 second), at which point this entire segment is extracted and processed by both machine learning models. After each successful prediction, the prediction_buffer is cleared, ensuring that subsequent predictions are based on fresh, contiguous audio, effectively creating a non-overlapping prediction window. This dual-buffer approach allows for simultaneous real-time visualization of the continuously incoming audio and periodic, non-overlapping analysis for classification, managed in a thread-safe manner using a `threading.Lock`.

---

### Inference Latency

The AST model had a latency of ~570ms while the RF classifier had a latency of only around 7ms. This makes sense when you think about the size of the model, and how much bigger the parameter space is for AST while also considering that it is trained on many more classes

---

### Model Analysis (Quickness and Stability)

The RF model was consistent and gave its confidence scores fairly quickly, but was not as stable as AST. AST consistently updated to predict the sound correctly. I believe this is due to AST having the ability to identify many more classes than the RF model.

---

### How Predictions and Confidence were visualized

The predictions and their corresponding confidence levels are dynamically presented within the `matplotlib` GUI's lower subplot using text rendering capabilities. A single `ax_text.text()` object is initialized, and its content is continuously updated within the `update_plot` function. For each prediction cycle, the current_ast_prediction and current_rf_prediction global variables, which hold formatted strings are used to construct the multi-line text displayed on the plot. This approach ensures that the user receives immediate, real-time feedback on what each model is classifying and how certain it is about that classification, directly integrated into the live visualization without requiring a separate UI framework.

---

### Different Environment Responses

The model performed poorly in a noisy room, as it always wanted to classify everything as the same class (in my case knock). And echoing room was not much better, although this was hard to test. The best classifications came from a quiet room. The room size itself didnt matter too much, but even background noise could throw the predicition off if it was loud enough.



