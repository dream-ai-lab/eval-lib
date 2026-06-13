"""Load an evaluation dataset exactly as pinned in the spec.

Kept separate from the runner so the contract layer (metrics, mlflow
logging) does not depend on heavy ML libraries being importable.
"""

from __future__ import annotations


def load_eval_split(spec):
    """Return ``(texts, refs)`` for the pinned dataset/split.

    ``spec.dataset`` defines hf_id, split, text_field, label_field, and an
    optional ``config`` (HF config name) and ``revision`` (a real HF git
    commit/tag). ``version`` is the human-facing pin recorded in the golden
    record; ``revision`` is what HF actually loads. In production always set
    ``revision`` to a commit SHA so the snapshot is byte-for-byte fixed.
    """
    from datasets import load_dataset

    ds = spec.dataset
    args = [ds.hf_id]
    config = getattr(ds, "config", None)
    if config:
        args.append(config)
    kwargs = {"split": ds.split}
    revision = getattr(ds, "revision", None)
    if revision:
        kwargs["revision"] = revision

    dataset = load_dataset(*args, **kwargs)
    texts = list(dataset[ds.text_field])
    refs = list(dataset[ds.label_field])
    return texts, refs
