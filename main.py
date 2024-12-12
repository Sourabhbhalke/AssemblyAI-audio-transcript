import os
import streamlit as st
from dotenv import load_dotenv
import assemblyai as aai

# Load API key from .env
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Streamlit UI
st.title("Podcast Analysis App")

# Audio file input
audio_url = st.text_input("Enter the audio file URL:", placeholder="e.g., https://assembly.ai/weaviate-podcast-109.mp3")

if audio_url:
    with st.spinner("Transcribing the audio..."):
        try:
            # Initialize AssemblyAI Transcriber
            transcriber = aai.Transcriber()

            # Step 2: Transcribe Audio
            config = aai.TranscriptionConfig(
                sentiment_analysis=True,
                speaker_labels=True
            )
            transcript = transcriber.transcribe(audio_url, config)

            # Display transcript
            st.subheader("Transcript")
            st.write(transcript.text)

            # Step 2: Count sentences
            sentences = transcript.get_sentences()
            st.write(f"The transcript has {len(sentences)} sentences.")

            # Step 3: Speaker Diarization and Sentiment Analysis
            st.subheader("Speaker and Sentiment Analysis")
            sentiments = {
                "POSITIVE": 0,
                "NEUTRAL": 0,
                "NEGATIVE": 0
            }

            for sentence in sentences:
                speaker = sentence.speaker or "Unknown Speaker"
                sentiment = sentence.sentiment
                sentiments[sentiment] += 1
                st.write(f"**{speaker}**: {sentence.text} ({sentiment})")

            st.write(f"\nSentiment Analysis Summary:")
            st.write(f"- Positive: {sentiments['POSITIVE']}")
            st.write(f"- Neutral: {sentiments['NEUTRAL']}")
            st.write(f"- Negative: {sentiments['NEGATIVE']}")

            # Step 4: Use LeMUR for summarization and Q&A
            st.subheader("LeMUR Analysis")

            # Prompt 1: Summarize
            prompt1 = "Provide a brief summary of the podcast."
            summary_result = transcript.lemur.task(prompt1, final_model=aai.LemurModel.claude3_5_sonnet)
            st.write("**Summary:**")
            st.write(summary_result.response)

            # Prompt 2: Question
            prompt2 = "Based on the transcript, what is agentic RAG?"
            question_result = transcript.lemur.task(prompt2, final_model=aai.LemurModel.claude3_5_sonnet)
            st.write("**Answer to 'What is agentic RAG?':**")
            st.write(question_result.response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
