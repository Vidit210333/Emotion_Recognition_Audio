import streamlit as st
import numpy as np
from PIL import Image
import io
import pickle  # Ensure to import pickle


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
        img = img.resize((64, 64))
        return img

    def recognize_emotions(self, frames, sampling_rate):
        emotions = []
        for i, frame in enumerate(frames):
            x = self.create_spectrogram_audio_break(frame, sampling_rate)
            x = image.img_to_array(x)
            x = x[:, :, :3]
            x = x.reshape((1,) + x.shape)  # Add batch dimension
            predictions = self.loaded_model.predict(x)
            predicted_label = self.class_to_labels[np.argmax(predictions)]
            emotions.append(predicted_label)
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

def main():
    st.title("Emotion Recognition from Audio")
    st.write("Upload an audio file and the application will predict the emotions expressed in the audio.")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav' if uploaded_file.type == 'audio/wav' else 'audio/mp3')

        with open("temp_audio_file.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the model
        model_path = 'model_checkpoint_Audio_Baseline_V2_optimized.pickle'  # Update with your model path
        model = pickle.load(open(model_path, 'rb'))

        recognizer = EmotionRecognizer(model)
        
        # Process the audio file
        frames, sampling_rate = recognizer.break_audio_into_frames("temp_audio_file.wav")
        
        # Recognize emotions
        emotions = recognizer.recognize_emotions(frames, sampling_rate)
        
        # Display the results
        st.write("Recognized Emotions:")
        for emotion in emotions:
            st.write(emotion)

if __name__ == "__main__":
    main()
