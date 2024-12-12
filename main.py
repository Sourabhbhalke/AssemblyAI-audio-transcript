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

            # Step 2: Transcribe Audio with Sentiment Analysis and Speaker Labels
            config = aai.TranscriptionConfig(
                sentiment_analysis=True,
                speaker_labels=True
            )
            transcript = transcriber.transcribe(audio_url, config)

            # Display transcript
            st.subheader("Transcript")
            st.text_area("Transcript", transcript.text, height=200, max_chars=None)

            # Step 2: Count sentences
            sentences = transcript.get_sentences()
            st.write(f"The transcript has {len(sentences)} sentences.")

            # Step 3: Speaker Diarization and Sentiment Analysis (2 tasks in 2 columns)
            st.subheader("Speaker and Sentiment Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Sentiment Analysis Summary (Part 1)**")
                sentiments = {
                    "POSITIVE": 0,
                    "NEUTRAL": 0,
                    "NEGATIVE": 0
                }

                # Limit to 5 lines of analysis in this part
                analysis_text = ""
                for sentence in sentences[:5]:  # First 5 sentences for part 1
                    speaker = sentence.speaker or "Unknown Speaker"
                    sentiment = getattr(sentence, 'sentiment', 'NEUTRAL')  # Default to 'NEUTRAL'
                    sentiments[sentiment] += 1
                    analysis_text += f"**{speaker}**: {sentence.text} ({sentiment})\n"

                st.text_area("Speaker Sentiment Analysis Part 1", analysis_text, height=200)

                st.write(f"\nSentiment Analysis Summary:")
                st.write(f"- Positive: {sentiments['POSITIVE']}")
                st.write(f"- Neutral: {sentiments['NEUTRAL']}")
                st.write(f"- Negative: {sentiments['NEGATIVE']}")

            with col2:
                st.write("**Sentiment Analysis Summary (Part 2)**")
                sentiments = {
                    "POSITIVE": 0,
                    "NEUTRAL": 0,
                    "NEGATIVE": 0
                }

                # Limit to 5 lines of analysis in this part
                analysis_text = ""
                for sentence in sentences[5:10]:  # Next 5 sentences for part 2
                    speaker = sentence.speaker or "Unknown Speaker"
                    sentiment = getattr(sentence, 'sentiment', 'NEUTRAL')  # Default to 'NEUTRAL'
                    sentiments[sentiment] += 1
                    analysis_text += f"**{speaker}**: {sentence.text} ({sentiment})\n"

                st.text_area("Speaker Sentiment Analysis Part 2", analysis_text, height=200)

                st.write(f"\nSentiment Analysis Summary:")
                st.write(f"- Positive: {sentiments['POSITIVE']}")
                st.write(f"- Neutral: {sentiments['NEUTRAL']}")
                st.write(f"- Negative: {sentiments['NEGATIVE']}")

            # Step 4: Use LeMUR for summarization and Q&A (2 tasks in 2 columns)
            st.subheader("LeMUR Analysis")
            col1, col2 = st.columns(2)

            with col1:
                prompt1 = "Provide a brief summary of the podcast."
                summary_result = transcript.lemur.task(prompt1, final_model=aai.LemurModel.claude3_5_sonnet)
                st.write("**Summary:**")
                st.text_area("Summary", summary_result.response, height=100, max_chars=None)

            with col2:
                prompt2 = "Based on the transcript, what is agentic RAG?"
                question_result = transcript.lemur.task(prompt2, final_model=aai.LemurModel.claude3_5_sonnet)
                st.write("**Answer to 'What is agentic RAG?':**")
                st.text_area("Answer", question_result.response, height=100, max_chars=None)

        except Exception as e:
            st.error(f"An error occurred: {e}")
