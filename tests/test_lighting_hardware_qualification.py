from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.qualify_lighting_hardware import (  # noqa: E402
    _device_signature,
    _run_component,
    summarize_qualification,
)


DEVICE = {
    "vendor": "Example Vendor",
    "renderer": "Example GPU",
    "opengl_version": "4.6",
    "glsl_version": "4.60",
}


def _fixture(device=DEVICE, *, passed=True):
    return {
        "device": device,
        "passed": passed,
        "viewpoints": {"exterior": {}, "interior": {}, "overview": {}},
        "dynamic_geometry_contract": {"passed": True},
        "packet_construction_contract": {"passed": True},
        "packet_local_light_ownership_contract": {"passed": True},
        "packet_receiver_contract": {"passed": True},
        "packet_alias_contract": {"passed": True},
    }


def _soak(devices=None, *, passed=True):
    devices = devices or [DEVICE, DEVICE, DEVICE]
    return {
        "passed": passed,
        "devices": devices,
        "results": [
            {
                "viewpoints": {"exterior": {}, "interior": {}, "overview": {}},
                "dynamic_geometry_contract": {"passed": True},
                "packet_construction_contract": {"passed": True},
                "packet_local_light_ownership_contract": {"passed": True},
                "packet_receiver_contract": {"passed": True},
                "packet_alias_contract": {"passed": True},
            }
            for _ in devices
        ],
    }


def _profile(device=DEVICE, *, passed=True):
    return {
        "device": device,
        "passed": passed,
        "generated_reference_budget": {"passed": passed},
    }


class LightingHardwareQualificationTests(unittest.TestCase):
    def _summary(self, *, fixture=None, soak=None, profile=None):
        artifact = ROOT / "artifacts" / "qualification-test"
        return summarize_qualification(
            fixture=_fixture() if fixture is None else fixture,
            soak=_soak() if soak is None else soak,
            profile=_profile() if profile is None else profile,
            fixture_report=artifact / "fixture" / "report.json",
            soak_report=artifact / "soak" / "report.json",
            profile_report=artifact / "profile.json",
        )

    def test_matching_component_reports_qualify_one_device(self) -> None:
        summary = self._summary()

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["device"], DEVICE)
        self.assertEqual(summary["device_consistency"]["expected_reports"], 5)
        self.assertEqual(summary["device_consistency"]["identified_reports"], 5)
        self.assertEqual(
            summary["components"]["generated_world_soak"]["viewpoints"],
            9,
        )

    def test_mixed_renderer_reports_are_rejected(self) -> None:
        different = {**DEVICE, "renderer": "Different GPU"}
        summary = self._summary(soak=_soak([DEVICE, different, DEVICE]))

        self.assertFalse(summary["device_consistency"]["consistent"])
        self.assertFalse(summary["passed"])

    def test_missing_identity_or_failed_component_is_rejected(self) -> None:
        missing_identity = self._summary(profile={"passed": True})
        failed_fixture = self._summary(fixture=_fixture(passed=False))

        self.assertFalse(missing_identity["device_consistency"]["passed"])
        self.assertFalse(missing_identity["passed"])
        self.assertFalse(failed_fixture["passed"])

    def test_missing_static_geometry_contract_is_rejected(self) -> None:
        fixture = _fixture()
        fixture.pop("dynamic_geometry_contract")
        soak = _soak()
        soak["results"][0]["dynamic_geometry_contract"] = {"passed": False}

        fixture_summary = self._summary(fixture=fixture)
        soak_summary = self._summary(soak=soak)

        self.assertFalse(fixture_summary["passed"])
        self.assertFalse(
            fixture_summary["components"]["fixture_visual_gate"]
            ["dynamic_geometry_contract_passed"]
        )
        self.assertFalse(soak_summary["passed"])
        self.assertFalse(
            soak_summary["components"]["generated_world_soak"]
            ["dynamic_geometry_contracts_passed"]
        )

    def test_missing_packet_alias_contract_is_rejected(self) -> None:
        fixture = _fixture()
        fixture.pop("packet_alias_contract")
        soak = _soak()
        soak["results"][0]["packet_alias_contract"] = {"passed": False}

        fixture_summary = self._summary(fixture=fixture)
        soak_summary = self._summary(soak=soak)

        self.assertFalse(fixture_summary["passed"])
        self.assertFalse(
            fixture_summary["components"]["fixture_visual_gate"]
            ["packet_alias_contract_passed"]
        )
        self.assertFalse(soak_summary["passed"])
        self.assertFalse(
            soak_summary["components"]["generated_world_soak"]
            ["packet_alias_contracts_passed"]
        )

    def test_missing_packet_construction_contract_is_rejected(self) -> None:
        fixture = _fixture()
        fixture.pop("packet_construction_contract")
        soak = _soak()
        soak["results"][0]["packet_construction_contract"] = {"passed": False}

        fixture_summary = self._summary(fixture=fixture)
        soak_summary = self._summary(soak=soak)

        self.assertFalse(fixture_summary["passed"])
        self.assertFalse(
            fixture_summary["components"]["fixture_visual_gate"]
            ["packet_construction_passed"]
        )
        self.assertFalse(soak_summary["passed"])
        self.assertFalse(
            soak_summary["components"]["generated_world_soak"]
            ["packet_construction_contracts_passed"]
        )

    def test_failed_packet_local_light_ownership_is_rejected(self) -> None:
        fixture = _fixture()
        fixture["packet_local_light_ownership_contract"] = {"passed": False}
        soak = _soak()
        soak["results"][0]["packet_local_light_ownership_contract"] = {
            "passed": False
        }

        fixture_summary = self._summary(fixture=fixture)
        soak_summary = self._summary(soak=soak)

        self.assertFalse(fixture_summary["passed"])
        self.assertFalse(
            fixture_summary["components"]["fixture_visual_gate"]
            ["packet_local_light_ownership_passed"]
        )
        self.assertFalse(soak_summary["passed"])
        self.assertFalse(
            soak_summary["components"]["generated_world_soak"]
            ["packet_local_light_ownership_contracts_passed"]
        )

    def test_device_signature_ignores_nonidentity_capabilities(self) -> None:
        with_limits = {**DEVICE, "max_texture_size": 32768}
        self.assertEqual(_device_signature(with_limits), _device_signature(DEVICE))

    def test_component_runner_rejects_stale_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            report_path.write_text('{"passed": true}\n', encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch(
                "scripts.qualify_lighting_hardware.subprocess.run",
                return_value=completed,
            ):
                report, execution = _run_component(["fake"], report_path)

        self.assertIsNone(report)
        self.assertTrue(execution["removed_stale_report"])
        self.assertFalse(execution["report_created"])
        self.assertFalse(execution["report_fresh"])

    def test_component_runner_accepts_report_written_by_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"

            def write_report(*_args, **_kwargs):
                report_path.write_text('{"passed": true}\n', encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "scripts.qualify_lighting_hardware.subprocess.run",
                side_effect=write_report,
            ):
                report, execution = _run_component(["fake"], report_path)

        self.assertEqual(report, {"passed": True})
        self.assertTrue(execution["report_fresh"])


if __name__ == "__main__":
    unittest.main()
