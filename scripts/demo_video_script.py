"""Curated demo runner for video narration with precomputed fast-path flow."""

from __future__ import annotations

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_flow import run_demo


def main() -> None:
    print("=" * 90)
    print("AyurYukti Demo Video Script | IndiaAI Innovation Challenge 2026")
    print("=" * 90)
    print("Before AyurYukti: manual notes, delayed coding, delayed outbreak recognition, static protocols.")
    print("After AyurYukti: voice-first EHR, evidence-backed recommendations, real-time surveillance, learning loop.")
    print()
    print("Narration cue: Scene 1 shows doctor speech becoming structured AHMIS-compatible EHR.")
    print("Narration cue: Scene 2 shows Prakriti + condition-aware ranked recommendations.")
    print("Narration cue: Scene 3 shows district-level anomaly alerts for early intervention.")
    print("Narration cue: Scene 4 shows YuktiShaala learning from outcomes.")
    print()
    start = time.perf_counter()
    run_demo()
    total = time.perf_counter() - start
    print()
    print(f"Video demo flow finished in {total:.1f}s (target < 180s).")


if __name__ == "__main__":
    main()
