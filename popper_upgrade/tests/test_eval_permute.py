import json
from pathlib import Path

import pandas as pd
import pytest

import popper_upgrade.runner as runner_module
from popper_upgrade.eval.permute import custom_factory
from popper_upgrade.runner import PopperRunner
from popper_upgrade.tests.fakes import FakePopper

REGISTRY = Path(__file__).resolve().parents[1] / "registry"


@pytest.fixture
def data_dir(tmp_path):
    frame = pd.DataFrame(
        {
            "text": [f"sentence {i} {'she' if i % 2 else 'he'}" for i in range(40)],
            "label": [i % 4 for i in range(40)],
            "kind": ["a" if i < 20 else "b" for i in range(40)],
        }
    )
    frame.to_csv(tmp_path / "t.csv", index=False)
    return tmp_path


def _frame(data_dir, seed, columns=("label",)):
    return custom_factory(str(data_dir), {"t": list(columns)}, seed)().table_dict["df_t"]


def test_same_seed_same_permutation(data_dir):
    first = _frame(data_dir, 7)
    again = _frame(data_dir, 7)

    pd.testing.assert_frame_equal(first, again)


def test_different_seed_different_permutation(data_dir):
    assert not _frame(data_dir, 7)["label"].equals(_frame(data_dir, 8)["label"])


def test_permutation_moves_only_the_chosen_columns(data_dir):
    original = pd.read_csv(data_dir / "t.csv")

    permuted = _frame(data_dir, 3)

    assert permuted["label"].tolist() != original["label"].tolist()
    assert sorted(permuted["label"]) == sorted(original["label"])
    pd.testing.assert_series_equal(permuted["text"], original["text"])
    pd.testing.assert_series_equal(permuted["kind"], original["kind"])


def test_multiple_columns_are_shuffled_together(data_dir):
    original = pd.read_csv(data_dir / "t.csv")
    pairs = set(zip(original["label"], original["kind"], strict=True))

    permuted = _frame(data_dir, 3, ("label", "kind"))

    assert set(zip(permuted["label"], permuted["kind"], strict=True)) == pairs
    assert permuted["label"].tolist() != original["label"].tolist()


def test_no_seed_leaves_data_untouched(data_dir):
    loader = custom_factory(str(data_dir), {"t": ["label"]}, None)()

    pd.testing.assert_frame_equal(loader.table_dict["df_t"], pd.read_csv(data_dir / "t.csv"))


def test_description_reflects_the_permuted_first_row(data_dir):
    original = pd.read_csv(data_dir / "t.csv")
    plain_desc = custom_factory(str(data_dir), {}, None)().data_desc
    seed = next(s for s in range(1, 50) if _frame(data_dir, s)["label"].iloc[0] != original["label"].iloc[0])

    loader = custom_factory(str(data_dir), {"t": ["label"]}, seed)()

    assert loader.data_desc != plain_desc
    assert loader.data_desc == loader._generate_data_description()


def test_each_call_builds_a_fresh_loader(data_dir):
    build = custom_factory(str(data_dir), {"t": ["label"]}, 3)

    assert build() is not build()


def test_factory_called_once_per_validate(monkeypatch):
    registered = []
    built = []

    class Recording(FakePopper):
        def register_data(self, data_path, loader_type="bio", **kwargs):
            registered.append(data_path)

    def factory():
        built.append(1)
        return object()

    monkeypatch.setattr(runner_module, "Popper", Recording)
    runner = PopperRunner(llm="claude-sonnet-4-5", data_loader_factory=factory)
    runner.register_data("ignored", loader_type="custom")
    runner.configure()

    runner.validate("H")
    runner.validate("H")

    assert len(built) == 2
    assert registered == []
    assert runner._popper.data_loader is not None


def test_registry_permutable_columns_are_label_columns_in_the_schema():
    datasets = json.loads((REGISTRY / "datasets.json").read_text(encoding="utf-8"))

    assert {d["name"]: d["permutable_columns"] for d in datasets} == {
        "winobias": ["subset"],
        "bbq": ["question"],
        "stereoset": ["label"],
    }
    for dataset in datasets:
        assert set(dataset["permutable_columns"]) <= set(dataset["schema"]["columns"])
