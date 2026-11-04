import json
import sys

import pytest

from scripts import benchmark_download_mirrors as benchmark


@pytest.mark.parametrize("domestic_available", [False, True])
def test_domestic_only_never_probes_or_falls_back_to_official(
    tmp_path, monkeypatch, domestic_available
):
    """An unreachable mirror must not cause a request to a blocked official host."""
    attempted = []
    official = benchmark.SOURCES["torch"]["official"]
    domestic = benchmark.SOURCES["torch"]["SJTU"]

    def measure(group, base, byte_limit, timeout):
        attempted.append(base)
        if base == official:
            pytest.fail("domestic-only contacted the official endpoint")
        if not domestic_available:
            raise OSError("mirror has no pinned CUDA wheel")
        return {"total_seconds": 1.0, "mib_per_second": 8.0}

    report_path = tmp_path / "report.json"
    env_path = tmp_path / "mirrors.env"
    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_download_mirrors.py", "--groups", "torch", "--domestic-only",
        "--rounds", "2", "--output", str(report_path), "--write-env", str(env_path),
    ])

    assert benchmark.main() == (0 if domestic_available else 1)
    assert attempted == [domestic, domestic]
    report = json.loads(report_path.read_text())
    assert report["domestic_only"] is True
    if domestic_available:
        assert report["selected"]["torch"] == domestic
        assert "MINIWAM_TORCH_INDEX=" in env_path.read_text()
    else:
        assert "torch" not in report["selected"]
        assert "MINIWAM_TORCH_INDEX=" not in env_path.read_text()


def test_prefer_domestic_keeps_official_fallback(tmp_path, monkeypatch):
    """The optional restriction must not change the existing general-use mode."""
    attempted = []
    official = benchmark.SOURCES["torch"]["official"]

    def measure(group, base, byte_limit, timeout):
        attempted.append(base)
        if base != official:
            raise OSError("mirror unavailable")
        return {"total_seconds": 1.0, "mib_per_second": 8.0}

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_download_mirrors.py", "--groups", "torch", "--prefer-domestic",
        "--rounds", "1", "--output", str(report_path),
    ])
    assert benchmark.main() == 0
    assert official in attempted
    assert json.loads(report_path.read_text())["selected"]["torch"] == official
