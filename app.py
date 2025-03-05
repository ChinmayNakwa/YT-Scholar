import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

transcribe_prompt = """You are a Youtube video summarizer. Summarize the following part of a transcript in bullet points: """
combine_summary_prompt = """Combine these summaries of different parts of a video into a single coherent summary with the most important points: """

question_prompt = """Based on this transcript excerpt, what important questions could be asked? Generate 2-3 thoughtful questions: """
combine_questions_prompt = """From the following sets of questions about different parts of a video, select and refine the 10 most important questions that cover the key concepts: """

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        
        return transcript
    
    except Exception as e:
        raise e

def chunk_text(text, max_chunk_size=3000):
    """Split text into chunks of approximately max_chunk_size tokens"""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_groq_content(text, prompt, model="llama3-70b-8192", max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt + text}
            ],
            temperature=0.5,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return f"Error generating content: {str(e)}"

def process_transcript_in_chunks(transcript, initial_prompt, combine_prompt, max_chunk_tokens=3000):
    # Split transcript into manageable chunks
    chunks = chunk_text(transcript, max_chunk_tokens)
    
    # Process each chunk
    results = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
            result = generate_groq_content(chunk, initial_prompt)
            results.append(result)
        progress_bar.progress((i + 1) / len(chunks))
    
    # If only one chunk, return it directly
    if len(results) == 1:
        return results[0]
    
    # Combine results from all chunks
    combined_results = "\n\n".join(results)
    final_result = generate_groq_content(combined_results, combine_prompt)
    
    return final_result

st.title("Youtube Transcript to Detailed Notes Convertor")
youtube_link = st.text_input("Enter Youtube Video Link:")

model_options = {
    "Llama 3 (70B)": "llama3-70b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768", 
    "Gemma 7B": "gemma-7b-it"
}

selected_model = st.selectbox("Select Model", list(model_options.keys()))

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes"):
    with st.spinner("Extracting transcript..."):
        transcript_text = extract_transcript_details(youtube_link)
    
    if transcript_text:
        model = model_options[selected_model]
        
        with st.spinner("Generating summary..."):
            summary = process_transcript_in_chunks(
                transcript_text, 
                transcribe_prompt, 
                combine_summary_prompt
            )
        
        with st.spinner("Generating questions..."):
            questions = process_transcript_in_chunks(
                transcript_text, 
                question_prompt, 
                combine_questions_prompt
            )
        
        st.markdown("## Detailed Notes:")
        st.write(summary)
        
        st.markdown("## Questions")
        st.write(questions)