# streamlit_app.py
import streamlit as st
from transformers import pipeline

# Step 1: Convert audio to text (ASR) using Whisper-small
def audio2text(uploaded_file):
    # Load Whisper-small model
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    # Save uploaded file to disk
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Run ASR directly on file path
    result = asr_model("temp.wav")
    return result["text"]

# Step 2: Run multi‑emotion analysis on transcript
def text2emotion(text):
    emotion_model = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )
    results = emotion_model(text)[0]
    # Sort emotions by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    # Return only top two
    return sorted_results[:2]

# Streamlit UI
def main():
    st.title("🎙️ Translating User Speech into Emotional Understanding for UBTECH Robotics")
    st.write("Upload an audio file (WAV/MP3/M4A) to transcribe and analyze emotions.")

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Step 1: Transcribe
        with st.spinner("Transcribing audio..."):
            text = audio2text(uploaded_file)
        st.subheader("Transcript")
        st.write(text)

        # Step 2: Emotion Analysis
        with st.spinner("Analyzing emotions..."):
            emotions = text2emotion(text)

        st.subheader("Top 2 Emotion Predictions")
        st.caption("Note: The model predicts only from 7 emotions — anger, disgust, fear, joy, neutral, sadness, surprise.")
        for emo in emotions:
            percent = emo['score'] * 100
            st.write(f"**{emo['label']}**: {percent:.2f}%")

if __name__ == "__main__":
    main()
