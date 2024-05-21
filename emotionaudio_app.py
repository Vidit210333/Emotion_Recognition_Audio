import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Google Drive file ID and destination file path
file_id = 'YOUR_FILE_ID'  # Replace with your file ID
output = 'model_checkpoint_Audio_Baseline_V2.keras'

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model(file_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return tf.keras.models.load_model(output)

# Load the model
model = download_and_load_model(file_id, output)

class EmotionRecognizer:

    def __init__(self, model):
        self.loaded_model = model
        self.class_to_labels = ["Fearful", "Neutral", "Happy", "Sad", "Angry"]

    def fig_to_image(self, fig):
        # Save the figure to a PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Convert PNG buffer to PIL Image
        img = Image.open(buf)

        return img

    def create_spectrogram_audio_break(self, y, sr):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)
        plt.close(fig)
        img = self.fig_to_image(fig)
        img = img.resize((224, 224))
        return img

    def recognize_emotions(self, frames, sampling_rate):
        emotions = []
        for i, frame in enumerate(frames):
            x = self.create_spectrogram_audio_break(frame, sampling_rate)
            x = np.array(x)[:, :, :3]  # Convert to numpy array and keep only RGB channels
            x = x.reshape((1,) + x.shape)  # Add batch dimension
            predictions = self.loaded_model.predict(x)
            for j, label in enumerate(self.class_to_labels):
                if predictions[0][j] == 1:
                    st.write(f"Frame {i} Emotion: {label}")  # Display emotion in Streamlit
                    emotions.append(label)
        return emotions

    def break_audio_into_frames(self, audio_file, frame_duration=3):
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        frame_length = sr * frame_duration
        frames = []
        for i in range(0, len(y), frame_length):
            frame = y[i:i+frame_length]
            frames.append(frame)

        return frames, sr

# Streamlit app
def main():
    st.title("Emotion Recognizer")

    recognizer = EmotionRecognizer(model)

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if audio_file is not None:
        frames, sampling_rate = recognizer.break_audio_into_frames(audio_file)
        emotions = recognizer.recognize_emotions(frames, sampling_rate)
        st.success("Emotion recognition completed.")

if __name__ == "__main__":
    main()
