"""Generate a tiny, redistributable synthetic corpus for a smoke-test run."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main(output: Path = Path("data/01_raw/audio"), clips_per_class: int = 12) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = 22_050
    seconds = 1.5
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    rng = np.random.default_rng(42)
    classes = {"low-tone": 220.0, "mid-tone": 660.0, "high-tone": 1760.0}
    for label, frequency in classes.items():
        for index in range(clips_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            signal = 0.2 * np.sin(2 * np.pi * frequency * time + phase)
            signal += 0.015 * rng.normal(size=len(time))
            path = output / f"{label}-{index:02d}.wav"
            sf.write(path, signal, sample_rate)
            path.with_suffix(".json").write_text(
                json.dumps({"tags": ["synthetic-demo", label]}), encoding="utf-8"
            )
    print(f"Generated {len(classes) * clips_per_class} demo clips in {output}")


if __name__ == "__main__":
    main()
