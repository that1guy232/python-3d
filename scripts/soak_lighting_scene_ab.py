"""Run the full-scene lighting A/B gate across deterministic generated worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_lighting_scene_ab.py"


def _result_summary(seed: int, output_dir: Path, report: dict) -> dict[str, object]:
    viewpoints = report.get("viewpoints", {})
    return {
        "seed": seed,
        "report": str((output_dir / "report.json").relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "passed": bool(report.get("passed")),
        "device": report.get("device"),
        "world": report.get("world"),
        "resource_counts": report.get("resource_counts"),
        "dynamic_geometry_contract": report.get("dynamic_geometry_contract"),
        "packet_receiver_contract": report.get("packet_receiver_contract"),
        "packet_alias_contract": report.get("packet_alias_contract"),
        "packet_construction_contract": report.get(
            "packet_construction_contract"
        ),
        "packet_local_light_ownership_contract": report.get(
            "packet_local_light_ownership_contract"
        ),
        "viewpoints": {
            name: {
                "legacy_drift_passed": bool(
                    details.get("legacy_drift", {}).get("passed")
                ),
                "packet_parity_passed": bool(
                    details.get("packet_parity", {}).get("passed")
                ),
                "packet_mean_absolute_error": details.get(
                    "packet_parity", {}
                ).get("mean_absolute_error"),
                "packet_p99_absolute_error": details.get(
                    "packet_parity", {}
                ).get("p99_absolute_error"),
                "packet_max_absolute_error": details.get(
                    "packet_parity", {}
                ).get("max_absolute_error"),
                "packet_changed_pixel_ratio": details.get(
                    "packet_parity", {}
                ).get("changed_pixel_ratio"),
            }
            for name, details in viewpoints.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for seed in args.seeds:
        seed_output = output_root / f"generated_{seed}"
        command = [
            sys.executable,
            str(CAPTURE_SCRIPT),
            "--world-mode",
            "generated",
            "--building-count",
            str(args.building_count),
            "--seed",
            str(seed),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--output-dir",
            str(seed_output),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        report_path = seed_output / "report.json"
        if completed.returncode != 0 or not report_path.exists():
            results.append(
                {
                    "seed": seed,
                    "passed": False,
                    "error": "capture process failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                }
            )
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results.append(_result_summary(seed, seed_output, report))

    aggregate = {
        "schema_version": 2,
        "world_mode": "generated",
        "building_count": args.building_count,
        "resolution": [args.width, args.height],
        "seeds": list(args.seeds),
        "devices": [
            result.get("device")
            for result in results
            if result.get("device") is not None
        ],
        "passed": len(results) == len(args.seeds)
        and all(bool(result.get("passed")) for result in results),
        "results": results,
    }
    (output_root / "report.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "lighting_scene_soak"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1201, 7349, 99173])
    parser.add_argument("--building-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.building_count < 1:
        raise SystemExit("--building-count must be at least 1")
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
