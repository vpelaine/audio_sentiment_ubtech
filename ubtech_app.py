# streamlit_app.py
import streamlit as st
from transformers import pipeline
import torchaudio
import io

# Step 1: Convert audio to text (ASR) using Whisper-small
def audio2text(audio_bytes):
    # Load Whisper-small model
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    # Load audio from uploaded file
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    # Whisper expects 16kHz mono audio
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    torchaudio.save("temp.wav", waveform, 16000)
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
    st.title("🎙️ Audio Emotion Analyzer")
    st.write("Upload an audio file (WAV/MP3) to transcribe and analyze emotions.")

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Step 1: Transcribe
        with st.spinner("Transcribing audio..."):
            text = audio2text(uploaded_file.read())
        st.subheader("Transcript")
        st.write(text)

        # Step 2: Emotion Analysis
        with st.spinner("Analyzing emotions..."):
            emotions = text2emotion(text)

        st.subheader("Top 2 Emotion Predictions")
        for emo in emotions:
            percent = emo['score'] * 100
            st.write(f"**{emo['label']}**: {percent:.2f}%")

if __name__ == "__main__":
    main()

