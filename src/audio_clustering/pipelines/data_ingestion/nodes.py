"""Audio Clustering — Data Ingestion Pipeline Nodes.

Loads audio files, validates duration, and extracts JSON metadata.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_audio_manifest(audio_dir: str, min_duration_sec: float = 0.5) -> pd.DataFrame:
    """Scan an audio directory and build a manifest DataFrame.

    Args:
        audio_dir: Path to directory containing .wav / .mp3 files.
        min_duration_sec: Minimum clip duration to include.

    Returns:
        DataFrame with columns: filepath, filename, duration_sec, tag (if JSON sidecar exists).
    """
    import librosa

    audio_dir_path = Path(audio_dir)
    records = []

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = [f for f in audio_dir_path.rglob("*") if f.suffix.lower() in audio_extensions]

    logger.info("Found %d audio files in %s", len(audio_files), audio_dir)

    for fpath in audio_files:
        try:
            duration = librosa.get_duration(path=str(fpath))
            if duration < min_duration_sec:
                logger.debug("Skipping %s (duration=%.2fs < %.2fs)", fpath.name, duration, min_duration_sec)
                continue

            record: dict[str, Any] = {
                "filepath": str(fpath),
                "filename": fpath.name,
                "stem": fpath.stem,
                "duration_sec": duration,
                "tag": None,
                "environment": None,
            }

            # Load JSON metadata sidecar if present (Freesound format)
            json_path = fpath.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    # Freesound JSON: tags is a list e.g. ["field-recording", "thunder", "thunderstorm"]
                    # Skip generic catch-all labels; pick most specific tag
                    _GENERIC = {"field-recording", "soundeffect", "sound-effects",
                                "fx", "sfx", "stereo", "mono", "sample"}
                    tags = meta.get("tags", [])
                    specific = [t for t in tags if t.lower() not in _GENERIC]
                    record["tag"] = specific[-1] if specific else (tags[-1] if tags else None)
                    record["all_tags"] = ",".join(tags)

            records.append(record)

        except Exception as exc:
            logger.warning("Could not process %s: %s", fpath.name, exc)

    df = pd.DataFrame(records)
    logger.info(
        "Manifest built: %d clips retained (%.1f%% with metadata)",
        len(df),
        100 * df["tag"].notna().mean() if len(df) > 0 else 0,
    )
    return df


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate manifest and log statistics.

    Args:
        manifest: Output of build_audio_manifest.

    Returns:
        Validated manifest (same rows, ensures required columns present).

    Raises:
        ValueError: If no valid audio files found.
    """
    required_cols = {"filepath", "filename", "duration_sec"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if len(manifest) == 0:
        raise ValueError(
            "No valid audio files found. Check 'audio_dir' in parameters "
            "and ensure files are longer than 'min_duration_sec'."
        )

    logger.info("=== Manifest Statistics ===")
    logger.info("Total clips: %d", len(manifest))
    logger.info("Duration — min: %.1fs | mean: %.1fs | max: %.1fs",
                manifest.duration_sec.min(),
                manifest.duration_sec.mean(),
                manifest.duration_sec.max())
    logger.info("Clips with ground-truth tag: %d (%.1f%%)",
                manifest.tag.notna().sum(),
                100 * manifest.tag.notna().mean())

    if manifest.tag.notna().any():
        tag_counts = manifest.tag.value_counts()
        logger.info("Tag distribution:\n%s", tag_counts.to_string())

    return manifest
