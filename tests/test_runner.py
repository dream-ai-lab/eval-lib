"""log_run (system of record) + run_paper delegation.

Exercised against a throwaway local MLflow file store so the golden-record
tags are actually asserted, not just the return value.
"""

import mlflow
import pytest
import yaml

import eval_lib.runner as runner
from eval_lib import log_run, run_paper


def _write_spec(path, *, primary="accuracy", secondary=None, experimental=None, target="accuracy"):
    spec = {
        "spec_version": "1.0.0",
        "paper_id": "test-paper",
        "task": "demo",
        "dataset": {
            "hf_id": "org/ds", "split": "test", "version": "1.0.0",
            "revision": "abc", "text_field": "t", "label_field": "l",
        },
        "model": {"hf_id": "org/m", "revision": "def"},
        "inference": {"seed": 1, "max_length": 8, "batch_size": 2},
        "metrics": {"primary": primary},
        "reproduce_target": {target: {"min": 0.0, "max": 1.0}},
    }
    if secondary:
        spec["metrics"]["secondary"] = secondary
    if experimental:
        spec["metrics"]["experimental"] = experimental
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    # mlflow 3 deprecates the file store; opt back in for a self-contained test.
    # A unique absolute URI per test avoids mlflow's global store cache colliding
    # across tests (a shared relative URI resolves differently as cwd changes).
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    root = tmp_path / "mlruns"
    root.mkdir()
    mlflow.set_tracking_uri(root.as_uri())
    yield
    mlflow.end_run()  # don't leak an active run into the next test


def _tags(run_id):
    return mlflow.get_run(run_id).data.tags


def test_log_run_byo_is_experimental(mlflow_local, tmp_path):
    spec = _write_spec(tmp_path / "s.yaml", primary="win_rate",
                       experimental=["win_rate"], target="win_rate")
    out = log_run(spec, {"win_rate": 0.62}, role="proposal",
                  code_fingerprint="sha256:deadbeef",
                  params={"judge": "gpt-4o"})
    t = _tags(out["run_id"])
    assert t["eval_tier"] == "experimental"
    assert t["code_fingerprint"] == "sha256:deadbeef"
    assert t["metric_lib_version"] == "byo"
    assert out["results"]["win_rate"] == 0.62


def test_log_run_blessed_is_standard(mlflow_local, tmp_path):
    spec = _write_spec(tmp_path / "s.yaml")
    out = log_run(spec, {"accuracy": 0.91}, metric_lib_version="0.3.0")
    t = _tags(out["run_id"])
    assert t["eval_tier"] == "standard"
    assert t["metric_lib_version"] == "0.3.0"
    assert "code_fingerprint" not in t


def test_run_paper_delegates_to_log_run(mlflow_local, tmp_path, monkeypatch):
    spec = _write_spec(tmp_path / "s.yaml")
    monkeypatch.setattr(runner, "load_eval_split", lambda spec: (["a", "b"], [1, 0]))
    out = run_paper(spec, lambda texts: [1, 0])
    t = _tags(out["run_id"])
    assert out["results"]["accuracy"] == 1.0
    assert out["reproduce_passed"] is True
    assert t["eval_tier"] == "standard"
    assert t["metric_lib_version"] == "0.3.0"


def test_run_paper_experimental_metric_tags_fingerprint(mlflow_local, tmp_path, monkeypatch):
    # 'hits' is not in the registry: declared experimental + computed as secondary.
    spec = _write_spec(tmp_path / "s.yaml", secondary=["hits"], experimental=["hits"])
    monkeypatch.setattr(runner, "load_eval_split", lambda spec: (["a", "b"], [1, 0]))
    out = run_paper(spec, lambda texts: [1, 0],
                    extra_metrics={"hits": lambda p, r: float(sum(a == b for a, b in zip(p, r)))})
    t = _tags(out["run_id"])
    assert t["eval_tier"] == "experimental"
    assert t["experimental_metrics"] == "hits"
    assert t["code_fingerprint"].startswith("sha256:")
    assert out["results"]["hits"] == 2.0
