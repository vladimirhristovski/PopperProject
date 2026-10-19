import json
from pathlib import Path

import pandas as pd
import pytest

from popper_upgrade.eval.benchmark import discoverybench_examples, gender_examples

STAGED = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "discoverybench"


def _metadata(domain, qids):
    return {
        "id": 0,
        "domain": domain,
        "datasets": [
            {
                "name": "d.csv",
                "description": "d",
                "columns": {"raw": [{"name": "x", "description": "x"}, {"name": "y", "description": "y"}]},
            }
        ],
        "queries": [[{"qid": qid, "question_type": "context", "question": "q"} for qid in qids]],
    }


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "bench"
    rows = []
    for task, domain, count in (("alpha", "biology", 3), ("beta", "economics", 2), ("gamma", "physics", 1)):
        folder = root / "discoverybench" / "real" / "test" / task
        folder.mkdir(parents=True)
        pd.DataFrame({"x": range(30), "y": [i * 2 for i in range(30)]}).to_csv(folder / "d.csv", index=False)
        (folder / "metadata_0.json").write_text(json.dumps(_metadata(domain, range(count))), encoding="utf-8")
        for qid in range(count):
            rows.append(
                {
                    "dataset": task,
                    "metadataid": 0,
                    "query_id": qid,
                    "gold_hypo": f"{task} hypothesis {qid}",
                    "non-trivially falsifiable": 1,
                }
            )
    (root / "answer_key").mkdir()
    pd.DataFrame(rows).to_csv(root / "answer_key" / "answer_key_real_cleaned_1.csv", index=False)
    return root


def test_filters_tasks_and_pairs_original_with_permuted(benchmark):
    examples = discoverybench_examples(str(benchmark), {"alpha", "gamma"})

    assert {example["task"] for example in examples} == {"alpha", "gamma"}
    assert len(examples) == 2 * (3 + 1)
    pairs = {(example["hypothesis"], example["permuted"], example["answer"]) for example in examples}
    assert ("alpha hypothesis 0", False, True) in pairs
    assert ("alpha hypothesis 0", True, False) in pairs
    assert {example["domain"] for example in examples if example["task"] == "gamma"} == {"physics"}


def test_per_task_limit(benchmark):
    examples = discoverybench_examples(str(benchmark), {"alpha", "beta"}, per_task=1)

    assert len(examples) == 4
    assert {example["task"] for example in examples} == {"alpha", "beta"}


def test_factories_permute_only_with_a_seed_and_deterministically(benchmark):
    examples = discoverybench_examples(str(benchmark), {"alpha"})
    original = next(e for e in examples if not e["permuted"])
    negative = next(e for e in examples if e["permuted"])

    plain = original["factory"](None)().table_dict["df_d"]
    first = negative["factory"](9)().table_dict["df_d"]
    again = negative["factory"](9)().table_dict["df_d"]
    other = negative["factory"](10)().table_dict["df_d"]

    assert plain["x"].tolist() == list(range(30))
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)
    assert not first.equals(plain)


@pytest.mark.skipif(not STAGED.exists(), reason="DiscoveryBench is not staged")
def test_discoverybench_filters_tasks():
    examples = discoverybench_examples(str(STAGED), {"requirements_engineering_for_ML_enabled_systems"})

    assert examples
    assert {example["task"] for example in examples} == {"requirements_engineering_for_ML_enabled_systems"}
    assert {example["answer"] for example in examples} == {True, False}


def test_gender_examples_pair_permuted_runs_with_one_unpermuted_run():
    hypotheses = [{"id": "h1", "text": "H one"}, {"id": "h2", "text": "H two"}]

    examples = gender_examples(hypotheses, "/data", {"bbq": ["question"], "winobias": ["subset"]})

    assert len(examples) == 4
    permuted = [e for e in examples if e["permuted"]]
    plain = [e for e in examples if not e["permuted"]]
    assert {e["answer"] for e in permuted} == {False}
    assert {e["answer"] for e in plain} == {None}
    assert {e["replicates"] for e in plain} == {1}
    assert all("replicates" not in e for e in permuted)
    assert {e["hypothesis_id"] for e in examples} == {"h1", "h2"}
    assert {e["domain"] for e in examples} == {"social science"}
