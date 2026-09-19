"""The loop's Claude agents: who they run for, and how the Scout's run is verified.

No agent is started here. These read the workflow files and run the plain shell steps
locally with a stubbed `gh`, so they need `node`, `jq`, `git` and `bash` on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CLAUDE_WORKFLOWS = sorted(WORKFLOWS.glob("claude-*.yml"))
# The events a person's own pushes and PRs fire. Loop agents must stand aside on them.
CODE_EVENTS = {"pull_request", "pull_request_target", "push"}
LOOP_BRANCH_GATE = {
    "pull_request": "startsWith(github.head_ref, 'claude/')",
    "pull_request_target": "startsWith(github.head_ref, 'claude/')",
    "push": "startsWith(github.ref_name, 'claude/')",
}


def load(path: Path) -> dict[str, Any]:
    raw: dict[Any, Any] = yaml.safe_load(path.read_text())
    # YAML 1.1 reads a bare `on:` key as the boolean True.
    return {("on" if key is True else str(key)): value for key, value in raw.items()}


def triggers(doc: dict[str, Any]) -> set[str]:
    on = doc["on"]
    if isinstance(on, str):
        return {on}
    return set(on)


def test_the_loop_scripts_pass_their_own_tests() -> None:
    tests = sorted(str(p) for p in (ROOT / "scripts" / "tests").glob("*.test.mjs"))
    assert tests
    run = subprocess.run(["node", "--test", *tests], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr


def test_claude_agents_on_pushes_and_prs_run_only_for_loop_branches() -> None:
    gated = 0
    for path in CLAUDE_WORKFLOWS:
        doc = load(path)
        events = triggers(doc) & CODE_EVENTS
        if not events:
            continue
        gated += 1
        jobs: dict[str, Any] = doc["jobs"]
        for name, job in jobs.items():
            if job.get("needs"):
                continue  # skipped with the root job it needs
            condition = str(job.get("if", ""))
            for event in events:
                assert f"github.event_name != '{event}'" in condition, (path.name, name)
                assert LOOP_BRANCH_GATE[event] in condition, (path.name, name, event)
        # Every other job needs a gated job, so it is skipped (never red) with it -
        # unless its condition overrides that with always() or !cancelled().
        for name, job in jobs.items():
            condition = str(job.get("if", ""))
            assert "always()" not in condition and "cancelled()" not in condition, (path.name, name)
    assert gated >= 1, "the Auditor runs on pull_request, so at least one workflow is gated"


def test_the_auditor_skips_non_loop_prs() -> None:
    doc = load(WORKFLOWS / "claude-audit.yml")
    assert doc["jobs"]["audit"]["needs"] == "credentials"
    condition = doc["jobs"]["credentials"]["if"]
    assert "startsWith(github.head_ref, 'claude/')" in condition


def test_plain_ci_runs_on_every_push_and_pr() -> None:
    doc = load(WORKFLOWS / "ci.yml")
    assert {"push", "pull_request"} <= triggers(doc)
    for job in doc["jobs"].values():
        assert "if" not in job


def test_autonomous_building_stays_off() -> None:
    config = json.loads((ROOT / ".github" / "loop-config.json").read_text())
    assert config["autonomousBuildEnabled"] is False


def step(doc: dict[str, Any], job: str, name: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = doc["jobs"][job]["steps"]
    return next(s for s in steps if s.get("name") == name)


@pytest.mark.parametrize("path", CLAUDE_WORKFLOWS, ids=lambda p: p.name)
def test_no_token_stands_down_green(path: Path, tmp_path: Path) -> None:
    doc = load(path)
    credentials = doc["jobs"]["credentials"]
    script = next(s for s in credentials["steps"] if s.get("id") == "check")["run"]
    assert "${{" not in script
    output = tmp_path / "out"
    env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": "", "GITHUB_OUTPUT": str(output)}
    run = subprocess.run(["bash", "-eo", "pipefail", "-c", script], env=env, capture_output=True)
    assert run.returncode == 0
    assert "ready=false" in output.read_text()
    agents = 0
    for name, job in doc["jobs"].items():
        uses = [str(s.get("uses", "")) for s in job.get("steps", [])]
        if any(u.startswith("anthropics/claude-code-action") for u in uses):
            agents += 1
            assert "needs.credentials.outputs.ready == 'true'" in str(job.get("if", "")), name
    assert agents >= 1


VERIFY = "Verify Scout filed something or said why not"


@pytest.fixture
def verify_scout(tmp_path: Path) -> Any:
    """Run the Scout's verify step as GitHub would, with `gh` stubbed."""
    run_script = step(load(WORKFLOWS / "claude-scout.yml"), "scout", VERIFY)["run"]
    assert "${{" not in run_script

    # The step reads its checker from the commit the run started on.
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "scout-standdown.mjs", repo / "scripts")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-qm", "x"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # The agent's working tree is not trusted: a tampered copy there must not be used.
    (repo / "scripts" / "scout-standdown.mjs").write_text("console.log('ok');\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()

    def run(issues: list[int], final_text: str | None) -> subprocess.CompletedProcess[str]:
        gh = bin_dir / "gh"
        gh.write_text(f"#!/bin/sh\necho '{json.dumps([{'number': n} for n in issues])}'\n")
        gh.chmod(0o755)
        execution = tmp_path / "execution.json"
        if final_text is None:
            execution.unlink(missing_ok=True)
        else:
            execution.write_text(json.dumps([{"type": "result", "result": final_text}]))
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "x",
            "HIGH_WATER": "40",
            "EXECUTION_FILE": str(execution),
            "GITHUB_SHA": sha,
            "RUNNER_TEMP": str(runner_temp),
        }
        return subprocess.run(
            ["bash", "-eo", "pipefail", "-c", run_script],
            env=env,
            cwd=repo,
            capture_output=True,
            text=True,
        )

    return run


@pytest.mark.parametrize(
    "final",
    [
        "Done.\nSCOUT RESULT: nothing filed - every candidate was already covered",
        "Done.\nSCOUT-DECISION: none - every candidate was already covered",
        "Done.\nSCOUT-DECISION: None - covered",
        "Done.\nSCOUT-DECISION: NONE: covered",
        "Done.\nSCOUT-DECISION: filed 0",
        "Done.\n**SCOUT-DECISION: none - covered**",
    ],
)
def test_scout_verify_accepts_every_stand_down_phrasing(verify_scout: Any, final: str) -> None:
    run = verify_scout([12, 40], final)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "::notice::Scout filed nothing this run and said why" in run.stdout


@pytest.mark.parametrize(
    "final",
    [None, "I started four researchers and will wait for them.", "SCOUT-DECISION: filed 2", "ok"],
)
def test_scout_verify_fails_a_silent_empty_run(verify_scout: Any, final: str | None) -> None:
    run = verify_scout([12, 40], final)
    assert run.returncode == 1
    assert "::error::Scout filed ZERO issues" in run.stdout


def test_scout_verify_passes_when_something_was_filed(verify_scout: Any) -> None:
    run = verify_scout([12, 41, 42], None)
    assert run.returncode == 0
    assert "Scout filed 2 new proposal(s)." in run.stdout
