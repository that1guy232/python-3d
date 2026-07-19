"""Run the complete packet-lighting qualification suite on one GL device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_lighting_scene_ab.py"
SOAK_SCRIPT = ROOT / "scripts" / "soak_lighting_scene_ab.py"
PROFILE_SCRIPT = ROOT / "scripts" / "profile_packet_lighting.py"
DEVICE_IDENTITY_FIELDS = (
    "vendor",
    "renderer",
    "opengl_version",
    "glsl_version",
)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _device_signature(device: dict | None) -> tuple[str | None, ...] | None:
    if not device:
        return None
    return tuple(device.get(field) for field in DEVICE_IDENTITY_FIELDS)


def _view_count(report: dict | None) -> int:
    return len((report or {}).get("viewpoints", {}))


def _soak_view_count(report: dict | None) -> int:
    return sum(
        len(result.get("viewpoints", {}))
        for result in (report or {}).get("results", ())
    )


def summarize_qualification(
    *,
    fixture: dict | None,
    soak: dict | None,
    profile: dict | None,
    fixture_report: Path,
    soak_report: Path,
    profile_report: Path,
) -> dict[str, object]:
    """Combine component reports and reject mixed-device evidence."""

    fixture_device = (fixture or {}).get("device")
    soak_devices = tuple((soak or {}).get("devices", ()))
    profile_device = (profile or {}).get("device")
    signatures = [
        signature
        for signature in (
            _device_signature(fixture_device),
            *(_device_signature(device) for device in soak_devices),
            _device_signature(profile_device),
        )
        if signature is not None
    ]
    device_consistent = bool(signatures) and all(
        signature == signatures[0] for signature in signatures[1:]
    )
    expected_device_reports = 2 + len((soak or {}).get("results", ()))
    identity_complete = len(signatures) == expected_device_reports
    device_passed = device_consistent and identity_complete

    components = {
        "fixture_visual_gate": {
            "report": _display_path(fixture_report),
            "passed": bool((fixture or {}).get("passed"))
            and bool(
                (fixture or {}).get("dynamic_geometry_contract", {}).get("passed")
            )
            and bool(
                (fixture or {})
                .get("packet_construction_contract", {})
                .get("passed")
            )
            and bool(
                (fixture or {})
                .get("packet_local_light_ownership_contract", {})
                .get("passed")
            )
            and bool(
                (fixture or {}).get("packet_receiver_contract", {}).get("passed")
            )
            and bool(
                (fixture or {}).get("packet_alias_contract", {}).get("passed")
            ),
            "viewpoints": _view_count(fixture),
            "dynamic_geometry_contract_passed": bool(
                (fixture or {}).get("dynamic_geometry_contract", {}).get("passed")
            ),
            "packet_construction_passed": bool(
                (fixture or {})
                .get("packet_construction_contract", {})
                .get("passed")
            ),
            "packet_local_light_ownership_passed": bool(
                (fixture or {})
                .get("packet_local_light_ownership_contract", {})
                .get("passed")
            ),
            "packet_receiver_contract_passed": bool(
                (fixture or {}).get("packet_receiver_contract", {}).get("passed")
            ),
            "packet_alias_contract_passed": bool(
                (fixture or {}).get("packet_alias_contract", {}).get("passed")
            ),
        },
        "generated_world_soak": {
            "report": _display_path(soak_report),
            "passed": bool((soak or {}).get("passed"))
            and bool((soak or {}).get("results"))
            and all(
                bool(result.get("dynamic_geometry_contract", {}).get("passed"))
                for result in (soak or {}).get("results", ())
            )
            and all(
                bool(
                    result.get("packet_construction_contract", {}).get("passed")
                )
                for result in (soak or {}).get("results", ())
            )
            and all(
                bool(
                    result.get("packet_local_light_ownership_contract", {}).get(
                        "passed"
                    )
                )
                for result in (soak or {}).get("results", ())
            )
            and all(
                bool(
                    result.get("packet_receiver_contract", {}).get("passed")
                )
                for result in (soak or {}).get("results", ())
            )
            and all(
                bool(result.get("packet_alias_contract", {}).get("passed"))
                for result in (soak or {}).get("results", ())
            ),
            "worlds": len((soak or {}).get("results", ())),
            "viewpoints": _soak_view_count(soak),
            "dynamic_geometry_contracts_passed": bool(
                (soak or {}).get("results")
            )
            and all(
                bool(result.get("dynamic_geometry_contract", {}).get("passed"))
                for result in (soak or {}).get("results", ())
            ),
            "packet_construction_contracts_passed": bool(
                (soak or {}).get("results")
            )
            and all(
                bool(
                    result.get("packet_construction_contract", {}).get("passed")
                )
                for result in (soak or {}).get("results", ())
            ),
            "packet_local_light_ownership_contracts_passed": bool(
                (soak or {}).get("results")
            )
            and all(
                bool(
                    result.get("packet_local_light_ownership_contract", {}).get(
                        "passed"
                    )
                )
                for result in (soak or {}).get("results", ())
            ),
            "packet_receiver_contracts_passed": bool(
                (soak or {}).get("results")
            )
            and all(
                bool(
                    result.get("packet_receiver_contract", {}).get("passed")
                )
                for result in (soak or {}).get("results", ())
            ),
            "packet_alias_contracts_passed": bool(
                (soak or {}).get("results")
            )
            and all(
                bool(result.get("packet_alias_contract", {}).get("passed"))
                for result in (soak or {}).get("results", ())
            ),
        },
        "packet_scan_profile": {
            "report": _display_path(profile_report),
            "passed": bool((profile or {}).get("passed")),
            "generated_reference_budget": (profile or {}).get(
                "generated_reference_budget"
            ),
        },
    }
    components_passed = all(
        bool(component["passed"]) for component in components.values()
    )
    return {
        "device": profile_device or fixture_device,
        "device_consistency": {
            "identity_fields": list(DEVICE_IDENTITY_FIELDS),
            "expected_reports": expected_device_reports,
            "identified_reports": len(signatures),
            "consistent": device_consistent,
            "passed": device_passed,
        },
        "components": components,
        "passed": components_passed and device_passed,
    }


def _run_component(command: list[str], report_path: Path) -> tuple[dict | None, dict]:
    removed_stale_report = report_path.exists()
    if removed_stale_report:
        report_path.unlink()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report_created = report_path.exists()
    execution = {
        "command": command,
        "returncode": completed.returncode,
        "removed_stale_report": removed_stale_report,
        "report_created": report_created,
        "report_fresh": report_created,
    }
    if completed.returncode != 0 or not report_created:
        execution["stdout_tail"] = completed.stdout[-4000:]
        execution["stderr_tail"] = completed.stderr[-4000:]
        return None, execution
    return json.loads(report_path.read_text(encoding="utf-8")), execution


def qualify(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    fixture_dir = output_dir / "fixture"
    soak_dir = output_dir / "generated_soak"
    profile_report = output_dir / "packet_profile.json"
    fixture_report = fixture_dir / "report.json"
    soak_report = soak_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_command = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        "--output-dir",
        str(fixture_dir),
        "--width",
        str(args.visual_width),
        "--height",
        str(args.visual_height),
        "--seed",
        str(args.fixture_seed),
    ]
    soak_command = [
        sys.executable,
        str(SOAK_SCRIPT),
        "--output-dir",
        str(soak_dir),
        "--width",
        str(args.visual_width),
        "--height",
        str(args.visual_height),
        "--building-count",
        str(args.building_count),
        "--seeds",
        *(str(seed) for seed in args.seeds),
    ]
    profile_command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--output",
        str(profile_report),
        "--width",
        str(args.profile_width),
        "--height",
        str(args.profile_height),
        "--warmup-frames",
        str(args.profile_warmup_frames),
        "--frames",
        str(args.profile_samples),
        "--draws-per-sample",
        str(args.profile_draws_per_sample),
    ]

    fixture, fixture_execution = _run_component(
        fixture_command,
        fixture_report,
    )
    soak, soak_execution = _run_component(soak_command, soak_report)
    profile, profile_execution = _run_component(
        profile_command,
        profile_report,
    )
    summary = summarize_qualification(
        fixture=fixture,
        soak=soak,
        profile=profile,
        fixture_report=fixture_report,
        soak_report=soak_report,
        profile_report=profile_report,
    )
    report = {
        "schema_version": 1,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "configuration": {
            "visual_resolution": [args.visual_width, args.visual_height],
            "fixture_seed": args.fixture_seed,
            "generated_seeds": list(args.seeds),
            "building_count": args.building_count,
            "profile_resolution": [args.profile_width, args.profile_height],
            "profile_samples": args.profile_samples,
            "profile_draws_per_sample": args.profile_draws_per_sample,
        },
        **summary,
        "executions": {
            "fixture": fixture_execution,
            "generated_soak": soak_execution,
            "packet_profile": profile_execution,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "artifacts" / "lighting_hardware_qualification" / "current_device"
        ),
    )
    parser.add_argument("--visual-width", type=int, default=800)
    parser.add_argument("--visual-height", type=int, default=450)
    parser.add_argument("--fixture-seed", type=int, default=7349)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1201, 7349, 99173])
    parser.add_argument("--building-count", type=int, default=3)
    parser.add_argument("--profile-width", type=int, default=640)
    parser.add_argument("--profile-height", type=int, default=360)
    parser.add_argument("--profile-warmup-frames", type=int, default=3)
    parser.add_argument("--profile-samples", type=int, default=12)
    parser.add_argument("--profile-draws-per-sample", type=int, default=20)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.visual_width,
        args.visual_height,
        args.profile_width,
        args.profile_height,
        args.profile_warmup_frames,
        args.profile_samples,
        args.profile_draws_per_sample,
        args.building_count,
    ) < 1:
        raise SystemExit("resolutions, counts, and profile samples must be positive")
    report = qualify(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
