import streamlit as st
from google import genai
from google.genai import types
import os
from datetime import datetime
import asyncio
import sounddevice as sd
import numpy as np
import queue
import threading
import wave
import io

# Page configuration
st.set_page_config(page_title="Gemini AI Assistant", page_icon="🤖", layout="wide")

# Add audio configurations
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback for audio recording"""
    audio_queue.put(indata.copy())

# Initialize session state for message history and system instruction
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_instruction" not in st.session_state:
    st.session_state.system_instruction = ""
if "recording" not in st.session_state:
    st.session_state.recording = False
if "live_session" not in st.session_state:
    st.session_state.live_session = None

# Enhanced sidebar configuration
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    # Basic settings
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        top_p = st.slider("Top P", 0.0, 1.0, 0.95)
        top_k = st.slider("Top K", 1, 40, 20)
        max_output_tokens = st.slider("Max Output Tokens", 100, 1000, 100)
        stop_sequence = st.text_input("Stop Sequence", "STOP!")
        presence_penalty = st.slider("Presence Penalty", 0.0, 1.0, 0.0)
        frequency_penalty = st.slider("Frequency Penalty", 0.0, 1.0, 0.0)
    
    # System instruction
    system_instruction = st.text_area("System Instruction", 
                                    st.session_state.system_instruction,
                                    help="Set a system-level instruction for the AI")
    if system_instruction != st.session_state.system_instruction:
        st.session_state.system_instruction = system_instruction

    # Safety settings
    st.subheader("Safety Settings")
    safety_settings = [types.SafetySetting(
        category='HARM_CATEGORY_HATE_SPEECH',
        threshold='BLOCK_ONLY_HIGH',
    )]

# Main chat interface
st.title("💬 Gemini AI Assistant")

# Voice assistant controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Start Recording"):
        st.session_state.recording = True
        audio_queue.queue.clear()

with col2:
    if st.button("⏹️ Stop Recording"):
        st.session_state.recording = False

# Live API implementation
async def process_voice_input():
    if api_key:
        client = genai.Client(api_key=api_key)
        config = {
            'response_modalities': ['AUDIO', 'TEXT'],
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
        }

        async def audio_stream():
            while st.session_state.recording:
                if not audio_queue.empty():
                    data = audio_queue.get()
                    yield data.tobytes()

        async with client.aio.live.connect(
            model='gemini-2.0-flash-exp',
            config=config
        ) as session:
            st.session_state.live_session = session
            async for response in session.start_stream(
                stream=audio_stream(),
                mime_type='audio/pcm'
            ):
                if response.text:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.text}
                    )
                if response.audio:
                    # Play audio response
                    audio_data = np.frombuffer(response.audio.data, dtype=np.float32)
                    sd.play(audio_data, SAMPLE_RATE)
                    sd.wait()

# Updated API configuration and chat logic
if api_key:
    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Generate response using the new client interface
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=types.Part.from_text(prompt),
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    candidate_count=1,
                    max_output_tokens=max_output_tokens,
                    stop_sequences=[stop_sequence],
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    system_instruction=system_instruction if system_instruction else None,
                    safety_settings=safety_settings
                )
            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

else:
    st.warning("Please enter your Gemini API key in the sidebar to start.")

# Audio recording handler
if st.session_state.recording:
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback
    ):
        st.info("🎤 Recording... Click 'Stop Recording' when finished")
        asyncio.run(process_voice_input())

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Google Gemini AI")
