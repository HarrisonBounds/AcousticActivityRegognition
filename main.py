import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time

# --- Configuration ---
SAMPLE_RATE = 44100 
BLOCK_SIZE = 1024    
WINDOW_LENGTH_SECONDS = 0.05 
DISPLAY_WINDOW_SECONDS = 2   

# Calculate derived parameters
WINDOW_LENGTH_SAMPLES = int(SAMPLE_RATE * WINDOW_LENGTH_SECONDS)
DISPLAY_WINDOW_SAMPLES = int(SAMPLE_RATE * DISPLAY_WINDOW_SECONDS)

# Data buffer for displaying the waveform
audio_buffer = deque(maxlen=DISPLAY_WINDOW_SAMPLES)

# Lock for thread-safe access to the audio buffer
buffer_lock = threading.Lock()

# Flag to control the recording thread
recording_active = threading.Event()
recording_active.set() # Set to True initially

# --- Audio Callback Function ---
def audio_callback(indata, frames, time_info, status):
    """
    This function is called by sounddevice for each incoming audio block.
    """
    if status:
        print(status)
    if not recording_active.is_set():
        raise sd.CallbackStop 
    
    with buffer_lock:
        current_block = indata[:, 0] if indata.ndim > 1 else indata
        audio_buffer.extend(current_block)

    if len(audio_buffer) >= WINDOW_LENGTH_SAMPLES:
        with buffer_lock: 
            processing_window = np.array(list(audio_buffer)[-WINDOW_LENGTH_SAMPLES:])
            pass 

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(np.zeros(DISPLAY_WINDOW_SAMPLES), color='blue') # Initial empty plot
ax.set_ylim([-1, 1])  # Assuming audio is normalized between -1 and 1
ax.set_xlim([0, DISPLAY_WINDOW_SAMPLES])
ax.set_title("Real-time Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# --- Update function for FuncAnimation ---
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
        else:
            line.set_ydata(np.zeros(DISPLAY_WINDOW_SAMPLES))
    return line, 

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Recording audio at {SAMPLE_RATE} Hz with a block size of {BLOCK_SIZE} frames.")
    print(f"Displaying {DISPLAY_WINDOW_SECONDS} seconds of audio.")
    print("Press Ctrl+C to stop recording.")

    try:
        # Start the audio stream in a non-blocking way
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1, # Mono audio
            callback=audio_callback
        )

        ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)

        with stream:
            plt.show() # Call plt.show() to start the matplotlib event loop

    except KeyboardInterrupt:
        print("\nStopping recording...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        recording_active.clear() # Signal threads to stop
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        plt.close(fig) # Close the matplotlib window
        print("Recording stopped and resources released.")