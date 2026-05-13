"""
Module 04 -- Lesson 2: Text-to-Speech (TTS)
=============================================
The reverse of Whisper: text goes in, spoken audio comes out.

We use Windows' built-in SAPI (Speech API) -- no installs needed.
This is the same engine behind Cortana and Windows Narrator.

In production, you'd use:
  - ElevenLabs API  -- most natural voices, paid
  - Microsoft Azure TTS -- high quality, pay per character
  - Google Cloud TTS -- similar to Azure
  - Coqui TTS -- open source, runs locally

But for learning the concept, Windows SAPI works perfectly.

What we'll do:
  1. Basic text-to-speech
  2. Change voice, speed, and volume
  3. Save speech to a WAV file
  4. Build a text reader that speaks any file you give it
"""

import subprocess
import os
import time

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def speak_powershell(text, rate=0, voice_index=0):
    """
    Use Windows SAPI via PowerShell to speak text.
    rate: -10 (slowest) to 10 (fastest), 0 = normal
    voice_index: 0 = first installed voice, 1 = second, etc.
    """
    # PowerShell script that calls Windows Speech API
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices()
if ($voices.Count -gt {voice_index}) {{
    $synth.SelectVoice($voices[{voice_index}].VoiceInfo.Name)
}}
$synth.Rate = {rate}
$synth.Speak("{text.replace('"', "'")}")
"""
    subprocess.run(["powershell", "-Command", ps_script],
                   capture_output=True, timeout=30)


def save_speech_to_file(text, filename, rate=0, voice_index=0):
    """Save spoken text to a WAV file using Windows SAPI."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices()
if ($voices.Count -gt {voice_index}) {{
    $synth.SelectVoice($voices[{voice_index}].VoiceInfo.Name)
}}
$synth.Rate = {rate}
$synth.SetOutputToWaveFile("{filepath.replace(chr(92), '/')}")
$synth.Speak("{text.replace('"', "'")}")
$synth.SetOutputToDefaultAudioDevice()
"""
    subprocess.run(["powershell", "-Command", ps_script],
                   capture_output=True, timeout=30)
    return filepath


def list_voices():
    """List all installed voices on this Windows system."""
    ps_script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices()
$i = 0
foreach ($voice in $voices) {
    $info = $voice.VoiceInfo
    Write-Output "$i : $($info.Name) | $($info.Gender) | $($info.Culture)"
    $i++
}
"""
    result = subprocess.run(["powershell", "-Command", ps_script],
                            capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


# ==============================================================================
# PART 1: Basic TTS
# ==============================================================================

def part1_basic_tts():
    print("=" * 60)
    print("  PART 1: Basic Text-to-Speech")
    print("=" * 60)

    # List available voices
    print("\n  Available voices on your system:")
    voices = list_voices()
    for line in voices.split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    # Speak a simple message
    print("\n  Speaking: 'Hello! I am your AI assistant.'")
    speak_powershell("Hello! I am your AI assistant.")
    print("  Done.\n")


# ==============================================================================
# PART 2: Voice variations
# ==============================================================================

def part2_voice_variations():
    print("=" * 60)
    print("  PART 2: Voice Variations")
    print("=" * 60)

    text = "Artificial intelligence is changing the world."

    # Normal speed
    print(f"\n  Speaking at normal speed...")
    speak_powershell(text, rate=0)

    # Fast
    print(f"  Speaking fast (rate=5)...")
    speak_powershell(text, rate=5)

    # Slow
    print(f"  Speaking slow (rate=-3)...")
    speak_powershell(text, rate=-3)

    # Try second voice if available
    print(f"  Trying voice index 1...")
    speak_powershell(text, rate=0, voice_index=1)

    print("  Done.\n")


# ==============================================================================
# PART 3: Save to file
# ==============================================================================

def part3_save_to_file():
    print("=" * 60)
    print("  PART 3: Save Speech to WAV File")
    print("=" * 60)

    text = ("Welcome to the AI Writing Toolkit. "
            "This tool helps you rewrite, summarize, and translate text "
            "using artificial intelligence. Get started by pasting your text.")

    print(f"\n  Saving speech to WAV file...")
    filepath = save_speech_to_file(text, "tts_output.wav")

    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  Saved: tts_output.wav ({size:,} bytes)")
        print(f"  You can play this file with any audio player.")
    else:
        print(f"  Error: file not created")

    print()


# ==============================================================================
# PART 4: Interactive text reader
# ==============================================================================

def part4_interactive():
    print("=" * 60)
    print("  PART 4: Interactive Text Reader")
    print("=" * 60)
    print("\n  Type any text and the computer will speak it.")
    print("  Commands: /quit, /fast, /slow, /normal, /save")
    print()

    rate = 0

    while True:
        try:
            text = input("  You type: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text:
            continue
        if text == "/quit":
            break
        if text == "/fast":
            rate = 5
            print("  Speed set to fast")
            continue
        if text == "/slow":
            rate = -3
            print("  Speed set to slow")
            continue
        if text == "/normal":
            rate = 0
            print("  Speed set to normal")
            continue
        if text == "/save":
            last_text = text
            filepath = save_speech_to_file("This is a saved message", "tts_interactive.wav")
            print(f"  Saved to: tts_interactive.wav")
            continue

        print("  Speaking...", end=" ", flush=True)
        speak_powershell(text, rate=rate)
        print("Done.")

    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MODULE 04 -- LESSON 2: TEXT-TO-SPEECH")
    print("=" * 60 + "\n")

    part1_basic_tts()
    part2_voice_variations()
    part3_save_to_file()

    try:
        choice = input("  Try interactive text reader? (y/n): ").strip().lower()
        if choice == 'y':
            part4_interactive()
    except (KeyboardInterrupt, EOFError):
        pass

    print("=" * 60)
    print("  TTS complete! You now have both directions:")
    print("  Speech -> Text (Whisper)  and  Text -> Speech (SAPI)")
    print("  Next: combine them into a voice assistant!")
    print("=" * 60)


if __name__ == "__main__":
    main()
