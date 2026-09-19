"""The write-up's numbers must be the artifacts' numbers.

Every measured number in README.md's Results section, docs/RESULTS.md and the tables of
docs/RESUME_CLAIMS.md sits in a generated region. These tests regenerate each region
from the committed results/ files and fail on any difference, and fail if a measured
number is typed into the hand-written prose around the regions, where nothing would
catch it drifting.
"""

from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path

import pytest

from m5 import config, writeup

# Numbers that can only be results: decimals, percentages and thousands.
MEASURED = re.compile(r"\d\.\d|\d%|\d,\d{3}")


@pytest.fixture(scope="module")
def artifacts() -> writeup.Artifacts:
    return writeup.Artifacts.load(config.RESULTS_DIR)


def doc_text(doc: str) -> str:
    return (config.REPO_ROOT / doc).read_text()


def results_section(readme: str) -> str:
    match = re.search(r"^## Results\n(.*?)^## ", readme, re.DOTALL | re.MULTILINE)
    assert match, "README.md has no '## Results' section"
    return match[1]


@pytest.mark.parametrize("doc", writeup.DOCS)
def test_committed_docs_match_their_artifacts(doc: str, artifacts: writeup.Artifacts) -> None:
    text = doc_text(doc)
    assert writeup.render(text, artifacts) == text, f"{doc} is stale: run `uv run m5-writeup`"


def test_every_region_is_published(artifacts: writeup.Artifacts) -> None:
    used = {m["name"] for doc in writeup.DOCS for m in writeup.REGION.finditer(doc_text(doc))}
    assert used == set(writeup.REGIONS)


@pytest.mark.parametrize("doc", ["README.md", "docs/RESULTS.md"])
def test_prose_carries_no_measured_number(doc: str) -> None:
    text = doc_text(doc)
    prose = writeup.strip_regions(results_section(text) if doc == "README.md" else text)
    found = [line for line in prose.splitlines() if MEASURED.search(line)]
    assert not found, f"{doc}: move these numbers into a generated region: {found}"


def test_readme_shows_the_plots() -> None:
    section = results_section(doc_text("README.md"))
    for plot in ("forecast_vs_actual_CA_1.png", "shap_importance_CA_1.png"):
        assert f"](results/{plot})" in section
        assert (config.RESULTS_DIR / plot).is_file()


def test_a_changed_artifact_changes_the_docs(artifacts: writeup.Artifacts) -> None:
    metrics = copy.deepcopy(artifacts.metrics)
    metrics["models"]["xgboost"]["mean_wrmsse"] += 0.01
    moved = writeup.Artifacts(metrics, artifacts.shap, artifacts.snap, artifacts.manifest)
    text = doc_text("README.md")
    assert writeup.render(text, moved) != text


def test_check_mode_flags_drift_and_rewrite_fixes_it(tmp_path: Path) -> None:
    results = tmp_path / "results"
    shutil.copytree(config.RESULTS_DIR, results)
    for doc in writeup.DOCS:
        (tmp_path / doc).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(config.REPO_ROOT / doc, tmp_path / doc)
    args = ["--results-dir", str(results), "--repo-root", str(tmp_path)]
    assert writeup.main([*args, "--check"]) == 0

    snap = (results / "snap_lift.json").read_text()
    (results / "snap_lift.json").write_text(snap.replace('"lift": 0.118', '"lift": 0.128', 1))
    assert writeup.main([*args, "--check"]) == 1
    assert writeup.main(args) == 0
    assert "+12.8%" in (tmp_path / "docs/RESULTS.md").read_text()
    assert writeup.main([*args, "--check"]) == 0


@pytest.mark.parametrize(
    "text",
    [
        "<!-- GENERATED:nope:BEGIN -->\n<!-- GENERATED:nope:END -->",
        "<!-- GENERATED:headline:BEGIN -->\n",
        "<!-- GENERATED:headline:END -->",
        "<!-- GENERATED:headline:BEGIN -->\n<!-- GENERATED:scope:BEGIN -->\n"
        "<!-- GENERATED:scope:END -->\n<!-- GENERATED:headline:END -->",
    ],
)
def test_malformed_markers_are_refused(text: str, artifacts: writeup.Artifacts) -> None:
    with pytest.raises(ValueError):
        writeup.render(text, artifacts)


def test_prose_outside_regions_is_kept(artifacts: writeup.Artifacts) -> None:
    region = "<!-- GENERATED:headline:BEGIN -->\nSTALE BODY\n<!-- GENERATED:headline:END -->"
    text = f"intro\n\n{region}\n\nend\n"
    out = writeup.render(text, artifacts)
    assert out.startswith("intro\n\n") and out.endswith("\n\nend\n")
    assert "STALE BODY" not in out
    assert writeup.render(out, artifacts) == out


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (10, "10th"), (11, "11th"), (21, "21st")],
)
def test_ordinal(n: int, expected: str) -> None:
    assert writeup.ordinal(n) == expected
