"""Exercise installed-wheel inputs and research receipts outside the checkout."""
from hashlib import sha256
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from array_reconstruct.joint_virtual import reference_elements


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_geometry_preserves_registered_source_bytes_and_order():
    package = files("array_reconstruct").joinpath("data")
    for name in ("reference_element_geometry.csv", "reference_element_geometry_provenance.json"):
        assert package.joinpath(name).read_bytes() == (ROOT / "metadata" / name).read_bytes()
    provenance = json.loads(package.joinpath("reference_element_geometry_provenance.json").read_text())
    raw = package.joinpath("reference_element_geometry.csv").read_bytes()
    assert sha256(raw).hexdigest() == provenance["export_sha256"]
    default, actual_source = reference_elements(32)
    explicit, _ = reference_elements(32, ROOT / "metadata/reference_element_geometry.csv")
    assert default.shape == (1024, 2)
    assert_array_equal(default, explicit)
    assert actual_source["sha256"] == provenance["export_sha256"]


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    work = tmp_path_factory.mktemp("installed-wheel")
    source = work / "source"
    source.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copytree(ROOT / "src", source / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    wheels = work / "wheels"
    wheels.mkdir()
    subprocess.run([sys.executable, "-B", "-c",
                    "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
                    str(wheels)], cwd=source, check=True, capture_output=True, text=True)
    installed = work / "site-packages"
    with ZipFile(next(wheels.glob("*.whl"))) as archive:
        archive.extractall(installed)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE="1")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "4"

    def run(script, *arguments):
        return subprocess.run([sys.executable, "-B", "-c", script, *map(str, arguments)],
                              cwd=work, env=environment, check=True, capture_output=True, text=True)

    return work, installed, run


def test_wheel_reference_geometry_is_self_contained(installed_wheel):
    work, installed, run = installed_wheel
    output = work / "geometry.npz"
    run("from array_reconstruct.joint_virtual import reference_elements; import numpy as np, sys; "
        "xy, source = reference_elements(32); np.savez(sys.argv[1], xy=xy, sha256=source['sha256'])", output)
    with np.load(output, allow_pickle=False) as actual:
        expected, source = reference_elements(32)
        assert_array_equal(actual["xy"], expected)
        assert actual["sha256"].item() == source["sha256"]
    assert not (installed.parent / "metadata").exists()


def test_wheel_k4_api_records_unavailable_repository_launcher(installed_wheel):
    work, installed, run = installed_wheel
    policy = json.loads((ROOT / "configs/k4_development_20261112.json").read_text())
    policy["geometry"].update(array_side=3, scan_side=5)
    policy.update(seeds=[81312001], fault_states=[False], max_cases=1,
                  max_total_seconds=60., max_case_seconds=30.)
    policy["state_error_regimes"] = policy["state_error_regimes"][:1]
    policy_path = work / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf8")
    output = work / "k4-output"
    run("from array_reconstruct.k4_research import run_k4_research; import sys; "
        "run_k4_research(sys.argv[1], sys.argv[2], generate_operators=True, progress=lambda *a, **k: None)",
        output, policy_path)
    receipt = json.loads((output / "run_receipt.json").read_text())
    assert receipt["registered_case_count"] == receipt["completed_case_count"] == 1
    assert receipt["failed_case_count"] == receipt["timed_out_case_count"] == 0
    launcher = installed.parent / "scripts/run_k4_research.py"
    assert not launcher.exists()
    assert set(receipt["unavailable_sources"]) == {str(launcher)}
    assert str(launcher) not in receipt["source_hashes"]
    assert len(receipt["source_hashes"]) == 8
    for name, digest in receipt["source_hashes"].items():
        assert Path(name).is_relative_to(installed)
        assert sha256(Path(name).read_bytes()).hexdigest() == digest
