"""
Module 04 -- Lesson 3: Voice Assistant
========================================
The full loop:
  1. You speak into the mic
  2. Whisper transcribes your speech to text
  3. Ollama (local LLM) generates a response
  4. Windows SAPI speaks the response out loud
  5. Repeat

This is exactly how Alexa, Siri, and Google Assistant work:
  Speech-to-Text -> AI Brain -> Text-to-Speech

Except ours runs 100% locally -- no cloud, no internet, fully private.

Controls:
  Just speak when you see "Listening..."
  Say "goodbye" or "exit" to quit
  Press Ctrl+C to force quit
"""

import whisper
import sounddevice as sd
import numpy as np
import requests
import subprocess
import time

SAMPLE_RATE = 16000
LISTEN_DURATION = 5
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"


def transcribe(model, audio_np):
    """Transcribe audio numpy array to text using Whisper."""
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    audio_np = audio_np.astype(np.float32)
    audio_tensor = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio_tensor).to(model.device)
    options = whisper.DecodingOptions(fp16=False, language="en")
    result = whisper.decode(model, mel, options)
    return result.text.strip()


def ask_ollama(user_message, history):
    """Send a message to Ollama and get a response."""
    history.append({"role": "user", "content": user_message})
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": False,
            "messages": history
        }, timeout=120)
        reply = response.json()["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"Sorry, I could not connect to Ollama. Error: {e}"


def speak(text, rate=1):
    """Speak text using Windows SAPI."""
    if len(text) > 500:
        text = text[:500] + "... I will stop here to keep it brief."
    safe_text = text.replace('"', "'").replace("\n", " ").replace("\r", "")
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {rate}
$synth.Speak("{safe_text}")
"""
    try:
        subprocess.run(["powershell", "-Command", ps_script],
                       capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        pass


def listen():
    """Record audio from microphone."""
    audio = sd.rec(int(LISTEN_DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio


def main():
    print("\n" + "=" * 60)
    print("  MODULE 04 -- LESSON 3: VOICE ASSISTANT")
    print("  Speech -> Whisper -> Ollama -> TTS -> Speaker")
    print("  100% local, no internet needed")
    print("=" * 60)

    print("\n  Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("  Whisper ready.")

    print("  Checking Ollama connection...")
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        print("  Ollama ready.")
    except Exception:
        print("  WARNING: Ollama not running! Start it with 'ollama serve'")
        return

    history = [{
        "role": "system",
        "content": (
            "You are a friendly voice assistant. Keep responses SHORT -- "
            "2-3 sentences maximum. You are being spoken to and your response "
            "will be read aloud, so be conversational and concise. "
            "Never use bullet points, code blocks, or markdown."
        )
    }]

    greeting = "Hello! I am your voice assistant. Ask me anything."
    print(f"\n  Assistant: {greeting}")
    speak(greeting)

    print("\n  Say 'goodbye' or 'exit' to quit.")
    print("  " + "-" * 50)

    while True:
        try:
            print(f"\n  [Listening for {LISTEN_DURATION}s...] ", end="", flush=True)
            audio = listen()

            volume = np.abs(audio).mean()
            if volume < 0.001:
                print("(silence)")
                continue

            print("Transcribing... ", end="", flush=True)
            user_text = transcribe(whisper_model, audio)

            skip = ["", ".", "Thank you.", "Thanks for watching.",
                    "you", "You", "Thank you for watching.", "Bye."]
            if not user_text or user_text in skip:
                print(f"(no speech: '{user_text}')")
                continue

            print(f"\n  You said: {user_text}")

            if any(word in user_text.lower() for word in ["goodbye", "exit", "quit", "stop"]):
                farewell = "Goodbye! Have a great day."
                print(f"  Assistant: {farewell}")
                speak(farewell)
                break

            print("  Thinking... ", end="", flush=True)
            start = time.time()
            reply = ask_ollama(user_text, history)
            elapsed = time.time() - start
            print(f"({elapsed:.1f}s)")

            print(f"  Assistant: {reply}")
            speak(reply, rate=1)

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break

    print("\n" + "=" * 60)
    print("  Voice assistant session ended.")
    print("  You just built a fully local AI voice assistant!")
    print("=" * 60)


if __name__ == "__main__":
    main()
