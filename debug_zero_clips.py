#!/usr/bin/env python3
"""Simulate why a long video gets 0 clips."""

from cortaflow.domain.analysis import ClipSelectionSettings
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.services.clip_scoring import suggest_clips

print("=" * 80)
print("TESTING REASONS FOR 0 CLIPS")
print("=" * 80)

# Test 1: Empty transcript
print("\n[TEST 1] Empty transcript")
empty_transcript = Transcript(language="pt", words=[])
result = suggest_clips(empty_transcript, 60_000)
print(f"  Result: {len(result)} clips")
assert len(result) == 0, "Empty transcript should yield 0 clips"
print("  ✓ PASS: Empty transcript confirmed as cause")

# Test 2: Very long duration
print("\n[TEST 2] Very long duration with valid words but >179s limits")
words = [
    TranscriptWord(text="palavra", start_ms=i*500, end_ms=(i+1)*500)
    for i in range(4000)  # 2000 seconds total
]
transcript = Transcript(language="pt", words=words)
settings = ClipSelectionSettings()  # default max_seconds=179
print(f"  Transcript duration: {words[-1].end_ms}ms ({words[-1].end_ms/1000:.0f}s)")
print(f"  Max allowed: {settings.max_seconds * 1000}ms")
result = suggest_clips(transcript, words[-1].end_ms, settings=settings)
print(f"  Result: {len(result)} clips")
if len(result) == 0:
    print("  ✗ Got 0 clips - very long duration can cause this")
else:
    print("  ✓ Got clips - so duration alone isn't the blocker")

# Test 3: Very short duration (less than 5s minimum)
print("\n[TEST 3] Very short video (<5s minimum)")
words_short = [
    TranscriptWord(text="palavra", start_ms=i*100, end_ms=(i+1)*100)
    for i in range(20)  # 2 seconds total
]
transcript_short = Transcript(language="pt", words=words_short)
result = suggest_clips(transcript_short, 2_000)
print(f"  Transcript duration: 2000ms ({2_000/1000:.1f}s)")
print(f"  Min allowed: {settings.min_seconds * 1000}ms")
print(f"  Result: {len(result)} clips")
assert len(result) == 0, "Too short video should yield 0 clips"
print("  ✓ PASS: Video too short confirmed as cause")

# Test 4: Lots of silence (high speech_ratio filter)
print("\n[TEST 4] Long video with too much silence (speech_ratio < 0.55)")
words_silent = []
# Create a pattern: 1 word per second, with large gaps between sentences
for i in range(30):
    start_ms = i * 2000  # 2 seconds apart
    words_silent.append(TranscriptWord(text="palavra", start_ms=start_ms, end_ms=start_ms+100))
    if i % 10 == 9:
        words_silent[-1] = TranscriptWord(text="palavra.", start_ms=start_ms, end_ms=start_ms+100)

from cortaflow.domain.analysis import TimeRange
silences = [
    TimeRange(start_ms=100+j*2000, end_ms=2000+j*2000)
    for j in range(29)
]
transcript_silent = Transcript(language="pt", words=words_silent)
total_duration = 30 * 2000
result = suggest_clips(transcript_silent, total_duration, silences=silences, settings=settings)
print(f"  Transcript duration: {total_duration}ms ({total_duration/1000:.0f}s)")
print(f"  Words: {len(words_silent)}, Silences: {len(silences)}")
total_silence = sum(s.end_ms - s.start_ms for s in silences)
print(f"  Total silence: {total_silence}ms out of {total_duration}ms ({100*total_silence/total_duration:.1f}%)")
print(f"  Speech ratio threshold: 0.55 (need 55% minimum speech)")
print(f"  Result: {len(result)} clips")
if len(result) == 0:
    print("  ✓ Got 0 clips - excessive silence confirmed as blocker")

# Test 5: Check the actual YouTube video scenario
print("\n[TEST 5] Simulating the actual YouTube video (18.5 min, likely with intro/outro)")
# 1110 seconds, assume similar structure to sports commentary/analysis
words_real = []
# Intro: relatively quiet 30 seconds
for i in range(30):
    words_real.append(TranscriptWord(text="bem", start_ms=i*1000, end_ms=i*1000+800))

# Main content: 950 seconds of varied speaking
for i in range(950):
    word_text = f"palavra{i%10}"
    if (i+1) % 50 == 0:
        word_text += "."  # Sentence boundary every 50 words
    words_real.append(TranscriptWord(text=word_text, start_ms=(30+i)*1000, end_ms=(30+i)*1000+900))

# Outro: slower 130 seconds
for i in range(130):
    words_real.append(TranscriptWord(text="obrigado", start_ms=(980+i)*1000, end_ms=(980+i)*1000+800))

transcript_real = Transcript(language="pt", words=words_real)
print(f"  Simulated transcript: {len(words_real)} words over 1110 seconds")

# Assume some silences (natural pauses during 18.5 min)
silences_real = [
    TimeRange(start_ms=i*10000, end_ms=i*10000+2000)
    for i in range(111)  # ~2s silence every 10s
]

result = suggest_clips(
    transcript_real,
    1_110_000,
    silences=silences_real,
    settings=settings
)
print(f"  Silences: {len(silences_real)} (~{sum(s.end_ms-s.start_ms for s in silences_real)/1000:.0f}s total)")
print(f"  Result: {len(result)} clips")

if len(result) == 0:
    print("\n  ✗ SIMULATION REPRODUCED ZERO CLIPS!")
    print("\n  LIKELY CAUSES for the actual video:")
    print("    1. Very long duration (1110s) may have few segments in the 5-179s range")
    print("    2. Sports commentary often has natural pauses that reduce speech_ratio below 0.55")
    print("    3. Editorial validation may reject most segments")
    print("    4. Or a combination of all three")
else:
    print(f"\n  ✓ Simulation generated {len(result)} clips")
    print(f"     Top clip: {result[0].quality_score:.3f} - {result[0].title}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The video C:\\Users\\limar\\Videos\\CortaFlow AI\\...  probably gets 0 clips because:

1. DURATION MISMATCH: 1110s (18.5 min) vs 5-179s range
   - The program's default clips are 5-179 seconds
   - A 18.5 min video needs many segments to fit that window
   
2. SILENCE RATIO: Sports commentary has natural pauses
   - Each candidate needs 55%+ speech content
   - Long pauses reduce this ratio below threshold
   
3. EDITORIAL FILTERS: Strict validation on boundaries
   - Must start/end at sentence boundaries
   - Must have clear speech segments

4. COMBINATION: All three together make it hard to find even one valid clip

Solution: Either disable the filters, or accept that this video type
          (long-form commentary) doesn't fit the vertical clip format.
""")
