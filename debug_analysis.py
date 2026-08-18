#!/usr/bin/env python3
"""Debug script to trace why a specific video gets 0 clips."""

from pathlib import Path
from cortaflow.services.media_probe import probe_media
from cortaflow.services.transcription import FasterWhisperTranscriber
from cortaflow.services.scene_detection import detect_scenes, detect_silences
from cortaflow.services.audio_analysis import extract_audio_evidence
from cortaflow.services.clip_scoring import suggest_clips
from cortaflow.domain.analysis import ClipSelectionSettings
from cortaflow.services.editorial_validation import assess_word_range

video_path = Path(r"C:\Users\limar\Videos\CortaFlow AI\Palmeirenses CRITICAM João Adibe após ACORDO com Flamengo! Leila COMETE GAFE! Depay no Corinthians! [NtFs2ISQlp4].mp4")

print("=" * 80)
print("PIPELINE ANALYSIS DEBUG")
print("=" * 80)

print("\n[1/6] Probe media...")
metadata = probe_media(video_path)
print(f"  Duration: {metadata.duration_seconds}s ({metadata.duration_seconds/60:.1f} min)")
print(f"  Resolution: {metadata.width}x{metadata.height}")
total_ms = round(metadata.duration_seconds * 1000)

print("\n[2/6] Transcribe audio (this may take a minute)...")
transcriber = FasterWhisperTranscriber("small")
transcript = transcriber.transcribe(video_path)
print(f"  Words transcribed: {len(transcript.words)}")
if transcript.words:
    print(f"  Transcript duration: {transcript.words[-1].end_ms}ms ({transcript.words[-1].end_ms/1000:.1f}s)")
    print(f"  First 5 words: {' '.join(w.text for w in transcript.words[:5])}")
else:
    print("  ✗ NO WORDS IN TRANSCRIPT - Pipeline will return 0 clips immediately")

print("\n[3/6] Detect scenes...")
scenes = detect_scenes(video_path)
print(f"  Scene changes detected: {len(scenes)}")
if scenes:
    print(f"  First 3: {scenes[:3]}")

print("\n[4/6] Detect silences...")
silences = detect_silences(video_path)
print(f"  Silence segments: {len(silences)}")
if silences:
    total_silence_ms = sum(s.end_ms - s.start_ms for s in silences)
    print(f"  Total silence: {total_silence_ms}ms ({total_silence_ms/1000:.1f}s)")
    print(f"  First 3: {silences[:3]}")

print("\n[5/6] Extract audio evidence...")
audio_evidence = extract_audio_evidence(video_path, transcript, silences)
print(f"  Audio evidence points: {len(audio_evidence)}")

print("\n[6/6] Generate clip suggestions...")
settings = ClipSelectionSettings()
print(f"  Selection settings:")
print(f"    - min_seconds: {settings.min_seconds}")
print(f"    - max_seconds: {settings.max_seconds}")
print(f"    - max_results: {settings.max_results}")
print(f"    - selection_goal: {settings.selection_goal}")

suggestions = suggest_clips(
    transcript, 
    total_ms, 
    silences=silences, 
    scenes=scenes, 
    settings=settings, 
    audio_evidence=audio_evidence
)

print(f"\n  Result: {len(suggestions)} clip suggestions")

if suggestions:
    print("\n  ✓ Suggestions generated:")
    for i, s in enumerate(suggestions[:5], 1):
        print(f"    {i}. [{s.start_ms}ms-{s.end_ms}ms] ({s.duration_ms/1000:.1f}s)")
        print(f"       Title: {s.title}")
        print(f"       Score: {s.quality_score}")
        print(f"       Reason: {s.reason}")
else:
    print("\n  ✗ ZERO SUGGESTIONS - Debugging why...")
    
    if not transcript.words:
        print("\n  ROOT CAUSE: Empty transcript - no words to analyze")
    else:
        # Debug candidate generation
        print("\n  Analyzing first few candidates...")
        words = transcript.words
        minimum_ms = settings.min_seconds * 1000  # 5000ms
        maximum_ms = settings.max_seconds * 1000  # 179000ms
        
        candidates_checked = 0
        candidates_valid_duration = 0
        candidates_valid_speech = 0
        candidates_valid_editorial = 0
        
        # Sample first 10 start positions
        for start_idx in range(0, min(10, len(words))):
            start_word = words[start_idx]
            
            # Try a few end positions
            for end_idx in range(start_idx + 1, min(start_idx + 10, len(words))):
                end_word = words[end_idx]
                duration = end_word.end_ms - start_word.start_ms
                
                candidates_checked += 1
                
                # Check duration filter
                if duration < minimum_ms or duration > maximum_ms:
                    continue
                
                candidates_valid_duration += 1
                
                # Check speech ratio filter
                silence_overlap = sum(
                    max(0, min(end_word.end_ms, silence.end_ms) - max(start_word.start_ms, silence.start_ms))
                    for silence in silences
                )
                speech_ratio = max(0.0, 1 - silence_overlap / max(1, duration)) if duration > 0 else 0
                
                if speech_ratio < 0.55:
                    continue
                
                candidates_valid_speech += 1
                
                # Check editorial filter
                editorial = assess_word_range(words, start_idx, end_idx, silences)
                if not editorial.passes:
                    continue
                
                candidates_valid_editorial += 1
                
                # If we get here, this candidate should pass
                excerpt = " ".join(w.text for w in words[start_idx:end_idx+1])
                print(f"\n    Sample candidate that should pass:")
                print(f"      Range: [{start_idx}-{end_idx}]")
                print(f"      Duration: {duration}ms ({duration/1000:.1f}s) ✓ in range [{minimum_ms}-{maximum_ms}]")
                print(f"      Speech ratio: {speech_ratio:.2f} ✓ >= 0.55")
                print(f"      Editorial: passes={editorial.passes}, start_safe={editorial.start_safe}, end_safe={editorial.end_safe}")
                print(f"      Excerpt: {excerpt[:80]}...")
        
        print(f"\n  Filter analysis (sampled first 10 starts × 10 ends):")
        print(f"    - Candidates checked: {candidates_checked}")
        print(f"    - Valid duration: {candidates_valid_duration}")
        print(f"    - Valid speech ratio: {candidates_valid_speech}")
        print(f"    - Valid editorial: {candidates_valid_editorial}")
        
        if candidates_checked == 0:
            print("\n  ROOT CAUSE: Not enough words to create any candidates")
        elif candidates_valid_duration == 0:
            print(f"\n  ROOT CAUSE: All candidates outside duration range ({minimum_ms}ms-{maximum_ms}ms)")
        elif candidates_valid_speech == 0:
            print(f"\n  ROOT CAUSE: All candidates have speech_ratio < 0.55 (too much silence)")
        elif candidates_valid_editorial == 0:
            print(f"\n  ROOT CAUSE: All candidates fail editorial validation")

print("\n" + "=" * 80)
