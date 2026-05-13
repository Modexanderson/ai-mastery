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
  2. Transcribe it with Whisper
  3. Show timestamps and language detection
  4. Build a live dictation mode (talk and see text appear)
"""

import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import tempfile

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Whisper expects 16kHz mono audio
SAMPLE_RATE = 16000


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
    print("  Model loaded.\n")

    # Record audio from microphone
    duration = 5  # seconds
    print(f"  Recording {duration} seconds of audio...")
    print("  Speak now!\n")

    # sounddevice records audio as a NumPy array -- just like images are arrays!
    # Audio: 1D array of amplitude values over time
    # Images: 3D array of color values over space
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,       # mono
                   dtype='float32')
    sd.wait()  # wait until recording is done

    print(f"  Recording complete.")
    print(f"  Audio shape: {audio.shape}")
    print(f"  That's {audio.shape[0]:,} samples at {SAMPLE_RATE}Hz = {audio.shape[0]/SAMPLE_RATE:.1f}s")

    # Save the audio to a file
    audio_path = os.path.join(OUTPUT_DIR, "recording.wav")
    sf.write(audio_path, audio, SAMPLE_RATE)
    print(f"  Saved to: recording.wav\n")

    # Transcribe with Whisper
    print("  Transcribing with Whisper...")
    start = time.time()
    result = model.transcribe(audio_path)
    elapsed = time.time() - start

    print(f"\n  --- TRANSCRIPTION ---")
    print(f"  Text: {result['text']}")
    print(f"  Language: {result['language']}")
    print(f"  Time: {elapsed:.1f}s")

    # Show segments with timestamps
    if result['segments']:
        print(f"\n  --- SEGMENTS (with timestamps) ---")
        for seg in result['segments']:
            start_t = seg['start']
            end_t = seg['end']
            text = seg['text'].strip()
            print(f"  [{start_t:.1f}s -> {end_t:.1f}s] {text}")

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
            # Visual indicator
            print(f"  [Recording {chunk_duration}s...] ", end="", flush=True)

            # Record a chunk
            audio = sd.rec(int(chunk_duration * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype='float32')
            sd.wait()

            # Check if there's actual audio (not silence)
            volume = np.abs(audio).mean()
            if volume < 0.005:  # silence threshold
                print("(silence)")
                continue

            # Save to temp file for Whisper
            tmp_path = os.path.join(tempfile.gettempdir(), "whisper_chunk.wav")
            sf.write(tmp_path, audio, SAMPLE_RATE)

            # Transcribe
            result = model.transcribe(tmp_path, fp16=False)
            text = result['text'].strip()

            if text and text not in ["", ".", "Thank you.", "Thanks for watching."]:
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
# PART 3: Transcribe from a file
# ==============================================================================

def part3_transcribe_file(model):
    print("=" * 60)
    print("  PART 3: Transcribe Any Audio File")
    print("=" * 60)

    audio_path = os.path.join(OUTPUT_DIR, "recording.wav")
    if not os.path.exists(audio_path):
        print("\n  No recording.wav found. Run Part 1 first.")
        return

    print(f"\n  Transcribing: {audio_path}")

    # Whisper can transcribe any audio format: wav, mp3, m4a, etc.
    result = model.transcribe(audio_path, fp16=False)

    print(f"  Text: {result['text']}")
    print(f"  Language detected: {result['language']}")

    # You could also translate to English
    print("\n  Now translating to English (if not already)...")
    result_en = model.transcribe(audio_path, task="translate", fp16=False)
    print(f"  English: {result_en['text']}")
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
