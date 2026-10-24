"""Command-line interface for the audio transcriber.

Uses only the standard library (argparse) so ``--help`` and argument handling work
without the ML stack installed; heavy imports happen only when a job actually runs.

Examples
--------
    transcribe recording.m4a
    transcribe ./recordings --format srt --output-dir ./out
    transcribe interview.wav --language ja --num-speakers 2 --model large-v3
    transcribe meeting.mp3 --no-diarize --format all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber import __version__
from transcriber.config import MODEL_SIZES, SUPPORTED_LANGUAGES, Config
from transcriber.formats import FORMATS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Offline, multilingual (es/en/ja) conversation transcription "
                    "with speaker labels.",
    )
    p.add_argument("path", help="Audio file, or a folder of audio files.")
    p.add_argument("--language", choices=["auto", *SUPPORTED_LANGUAGES], default="auto",
                   help="Spoken language, or 'auto' to detect (default: auto).")
    p.add_argument("--model", choices=MODEL_SIZES, default="medium",
                   help="Whisper model size (default: medium).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Compute device (default: auto-detect).")
    p.add_argument("--compute-type", default=None,
                   help="CTranslate2 compute type, e.g. int8, float16 (default: by device).")
    diar = p.add_mutually_exclusive_group()
    diar.add_argument("--diarize", dest="diarize", action="store_true", default=True,
                      help="Label speakers (default).")
    diar.add_argument("--no-diarize", dest="diarize", action="store_false",
                      help="Plain transcript, no speaker labels.")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Exact speaker count hint for diarization.")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--format", choices=[*FORMATS, "all"], default="txt",
                   help="Output format, or 'all' (default: txt).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write outputs (default: next to each input file).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported here so --help / --version don't pull in heavy deps.
    from transcriber.pipeline import iter_audio_files, transcribe_file
    from transcriber.formats import render

    config = Config(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=None if args.language == "auto" else args.language,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    try:
        files = iter_audio_files(args.path)
    except FileNotFoundError:
        print(f"error: no such file or folder: {args.path}", file=sys.stderr)
        return 2
    if not files:
        print(f"error: no audio files found in {args.path}", file=sys.stderr)
        return 2

    formats = FORMATS if args.format == "all" else [args.format]
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        try:
            transcript = transcribe_file(
                f, config, progress=lambda m: print(f"    {m}", file=sys.stderr)
            )
        except Exception as exc:  # keep batch jobs going
            print(f"    FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        dest_dir = out_dir or f.parent
        for fmt in formats:
            dest = dest_dir / f"{f.stem}.{fmt}"
            dest.write_text(render(transcript, fmt), encoding="utf-8")
            print(f"    wrote {dest}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
