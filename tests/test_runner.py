"""log_run (system of record) + run_paper delegation.

The golden record now goes to Weights & Biases. We stub ``wandb.init`` with a
recording fake so the mapping (config / tags / logged metrics) is asserted
without a network or API key. A live offline smoke is run separately.
"""

import pytest
import yaml

import eval_lib.runner as runner
from eval_lib import __version__, log_run, run_paper


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


class FakeRun:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.config = dict(kwargs.get("config") or {})
        self.tags = list(kwargs.get("tags") or [])
        self.id = "run-fake"
        self.entity = kwargs.get("entity") or "ent"
        self.project = kwargs.get("project") or "proj"
        self.summary = {}
        self.logged = []
        self.saved = []
        self.finished = False

    def log(self, data):
        self.logged.append(dict(data))
        self.summary.update(data)

    def save(self, path, **kwargs):
        self.saved.append(path)

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    created = []

    def fake_init(**kwargs):
        run = FakeRun(**kwargs)
        created.append(run)
        return run

    monkeypatch.setattr(runner.wandb, "init", fake_init)
    return created


def test_log_run_byo_is_experimental(fake_wandb, tmp_path):
    spec = _write_spec(tmp_path / "s.yaml", primary="win_rate",
                       experimental=["win_rate"], target="win_rate")
    out = log_run(spec, {"win_rate": 0.62}, role="proposal",
                  code_fingerprint="sha256:deadbeef", params={"judge": "gpt-4o"})
    run = fake_wandb[-1]
    assert run.config["eval_tier"] == "experimental"
    assert run.config["code_fingerprint"] == "sha256:deadbeef"
    assert run.config["metric_lib_version"] == "byo"
    assert run.config["judge"] == "gpt-4o"          # params merged into config
    assert "tier:experimental" in run.tags
    assert run.logged[-1]["win_rate"] == 0.62       # results logged
    assert run.finished is True
    assert out["results"]["win_rate"] == 0.62


def test_log_run_blessed_is_standard(fake_wandb, tmp_path):
    spec = _write_spec(tmp_path / "s.yaml")
    out = log_run(spec, {"accuracy": 0.91}, metric_lib_version="0.4.0")
    run = fake_wandb[-1]
    assert run.config["eval_tier"] == "standard"
    assert run.config["metric_lib_version"] == "0.4.0"
    assert "code_fingerprint" not in run.config
    assert out["run_id"] == "run-fake"


def test_run_paper_delegates_to_log_run(fake_wandb, tmp_path, monkeypatch):
    spec = _write_spec(tmp_path / "s.yaml")
    monkeypatch.setattr(runner, "load_eval_split", lambda spec: (["a", "b"], [1, 0]))
    out = run_paper(spec, lambda texts: [1, 0])
    run = fake_wandb[-1]
    assert out["results"]["accuracy"] == 1.0
    assert out["reproduce_passed"] is True
    assert run.config["eval_tier"] == "standard"
    assert run.config["metric_lib_version"] == __version__   # blessed path
    assert run.init_kwargs["group"] == "test-paper"          # grouped by paper_id


def test_run_paper_experimental_metric_tags_fingerprint(fake_wandb, tmp_path, monkeypatch):
    # 'hits' is not in the registry: declared experimental + computed as secondary.
    spec = _write_spec(tmp_path / "s.yaml", secondary=["hits"], experimental=["hits"])
    monkeypatch.setattr(runner, "load_eval_split", lambda spec: (["a", "b"], [1, 0]))
    out = run_paper(spec, lambda texts: [1, 0],
                    extra_metrics={"hits": lambda p, r: float(sum(a == b for a, b in zip(p, r)))})
    run = fake_wandb[-1]
    assert run.config["eval_tier"] == "experimental"
    assert run.config["experimental_metrics"] == "hits"
    assert run.config["code_fingerprint"].startswith("sha256:")
    assert out["results"]["hits"] == 2.0
