# streamlit_app.py
import streamlit as st
from transformers import pipeline
import tempfile

# Step 1: Convert audio to text (ASR) using Whisper-small
def audio2text(uploaded_file):
    # Load Whisper-small model once per call
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Run ASR directly on file path
    result = asr_model(temp_path)
    return result["text"]

# Step 2: Run multi‑emotion analysis on transcript
def text2emotion(text):
    emotion_model = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )
    results = emotion_model(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:2]

# Streamlit UI
def main():
    st.title("🎙️ Translating User Speech into Emotional Understanding for UBTECH Robotics")
    st.write("Upload one or more audio files (WAV/MP3/M4A) to transcribe and analyze emotions.")

    uploaded_files = st.file_uploader(
        "Upload Audio Files",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.subheader(f"File: {uploaded_file.name}")
            st.audio(uploaded_file, format="audio/wav")

            # Step 1: Transcribe
            with st.spinner("Transcribing audio..."):
                text = audio2text(uploaded_file)
            st.write("### Transcript")
            st.write(text)

            # Step 2: Emotion Analysis
            with st.spinner("Analyzing emotions..."):
                emotions = text2emotion(text)

            st.write("### Top 2 Emotion Predictions")
            st.caption("Model predicts from: anger, disgust, fear, joy, neutral, sadness, surprise.")
            for emo in emotions:
                percent = emo['score'] * 100
                st.write(f"**{emo['label']}**: {percent:.2f}%")

if __name__ == "__main__":
    main()
