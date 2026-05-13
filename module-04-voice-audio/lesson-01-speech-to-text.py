"""
Module 04 -- Lesson 1: Speech-to-Text with Whisper
====================================================
Whisper is OpenAI's speech recognition model. It runs locally on your GPU.

How it works (simplified):
  1. Audio goes in as a waveform (just numbers over time, like pixels for images)
  2. The neural network processes the audio
  3. Text comes out -- what was said, with timestamps

Whisper model sizes:
  tiny    (~39M params)  -- fastest, least accurate
  base    (~74M params)  -- good balance for testing
  small   (~244M params) -- solid accuracy
  medium  (~769M params) -- very good
  large   (~1.5B params) -- best accuracy, needs more VRAM

Your RTX 4070 can run medium comfortably, large with some patience.

What we'll do:
  1. Record audio from your microphone
  2. Transcribe it with Whisper (bypassing ffmpeg -- direct numpy array)
  3. Show language detection
  4. Build a live dictation mode (talk and see text appear)
"""

import whisper
import sounddevice as sd
import numpy as np
import os
import time
import torch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Whisper expects 16kHz mono audio
SAMPLE_RATE = 16000


def transcribe_audio_array(model, audio_np):
    """
    Transcribe a numpy audio array directly with Whisper.
    This bypasses ffmpeg entirely -- we feed raw audio straight to the model.

    Steps (what Whisper does internally):
      1. Pad or trim audio to 30 seconds (Whisper's fixed window)
      2. Convert to a mel spectrogram (frequency representation)
      3. Feed through the neural network
      4. Decode the output tokens into text
    """
    # Ensure audio is 1D float32
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    audio_np = audio_np.astype(np.float32)

    # Pad or trim to 30 seconds (Whisper's expected input length)
    audio_tensor = whisper.pad_or_trim(audio_np)

    # Convert to mel spectrogram -- this is how audio is represented for neural nets
    # Like converting a photo to grayscale before edge detection
    mel = whisper.log_mel_spectrogram(audio_tensor).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    # Decode (transcribe)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return {"text": result.text, "language": language}


# ==============================================================================
# PART 1: Record and transcribe
# ==============================================================================

def part1_record_and_transcribe():
    print("=" * 60)
    print("  PART 1: Record & Transcribe")
    print("=" * 60)

    # Load Whisper model -- downloads on first run
    print("\n  Loading Whisper 'base' model (74M params)...")
    model = whisper.load_model("base")
    print(f"  Model loaded on: {model.device}")

    # Record audio from microphone
    duration = 5  # seconds
    print(f"\n  Recording {duration} seconds of audio...")
    print("  Speak now!\n")

    # sounddevice records audio as a NumPy array -- just like images are arrays!
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()

    print(f"  Recording complete.")
    print(f"  Audio shape: {audio.shape}")
    print(f"  That's {audio.shape[0]:,} samples at {SAMPLE_RATE}Hz = {audio.shape[0]/SAMPLE_RATE:.1f}s\n")

    # Transcribe directly from numpy array (no ffmpeg needed!)
    print("  Transcribing with Whisper...")
    start = time.time()
    result = transcribe_audio_array(model, audio)
    elapsed = time.time() - start

    print(f"\n  --- TRANSCRIPTION ---")
    print(f"  Text: {result['text']}")
    print(f"  Language: {result['language']}")
    print(f"  Time: {elapsed:.1f}s")

    print()
    return model


# ==============================================================================
# PART 2: Live dictation mode
# ==============================================================================

def part2_live_dictation(model):
    print("=" * 60)
    print("  PART 2: Live Dictation Mode")
    print("=" * 60)
    print("\n  Speak in chunks. After each chunk, Whisper transcribes it.")
    print("  Press Ctrl+C to stop.\n")

    chunk_duration = 4  # seconds per chunk
    full_transcript = []

    try:
        while True:
            print(f"  [Recording {chunk_duration}s...] ", end="", flush=True)

            # Record a chunk
            audio = sd.rec(int(chunk_duration * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype='float32')
            sd.wait()

            # Check if there's actual audio (not silence)
            volume = np.abs(audio).mean()
            if volume < 0.005:
                print("(silence)")
                continue

            # Transcribe directly from array
            result = transcribe_audio_array(model, audio)
            text = result['text'].strip()

            # Filter out Whisper hallucinations on near-silence
            skip_phrases = ["", ".", "Thank you.", "Thanks for watching.",
                            "you", "You", "Thank you for watching."]
            if text and text not in skip_phrases:
                full_transcript.append(text)
                print(f"{text}")
            else:
                print("(no speech detected)")

    except KeyboardInterrupt:
        pass

    # Show full transcript
    if full_transcript:
        print(f"\n\n  --- FULL TRANSCRIPT ---")
        print(f"  {' '.join(full_transcript)}")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MODULE 04 -- LESSON 1: SPEECH-TO-TEXT WITH WHISPER")
    print("=" * 60 + "\n")

    model = part1_record_and_transcribe()

    try:
        choice = input("  Try live dictation mode? (y/n): ").strip().lower()
        if choice == 'y':
            part2_live_dictation(model)
    except (KeyboardInterrupt, EOFError):
        pass

    print("=" * 60)
    print("  Done! Whisper can transcribe audio in 99 languages.")
    print("  Next lesson: Text-to-Speech (the reverse!)")
    print("=" * 60)


if __name__ == "__main__":
    main()
