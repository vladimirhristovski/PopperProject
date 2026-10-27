import functools
import json
import random
import sys
from pathlib import Path

from popper.benchmark import discovery_bench_hypothesis
from popper.utils import DiscoveryBenchDataLoader

from popper_upgrade.eval.permute import custom_factory

GENDER_TASK = "gender_bias"
GENDER_DOMAIN = "social science"


def discovery_factory(task_path, metadata_path, seed):
    def build():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8", errors="replace"))
        loader = DiscoveryBenchDataLoader(task_path, metadata)
        if seed is not None:
            loader.permute_selected_columns(random_seed=seed)
        return loader

    return build


def _entry_key(entry):
    return (entry["task"], entry["metadataid"], entry["query_id"])


def stratified_sample(entries, total, seed):
    by_task = {}
    for entry in entries:
        by_task.setdefault(entry["task"], []).append(entry)
    rng = random.Random(seed)
    quotas = dict.fromkeys(by_task, 0)
    open_tasks = sorted(by_task)
    remaining = min(total, len(entries))
    while remaining and open_tasks:
        share, extra = divmod(remaining, len(open_tasks))
        bonus = set(rng.sample(open_tasks, extra))
        remaining = 0
        for task in open_tasks:
            want = share + (task in bonus)
            take = min(want, len(by_task[task]) - quotas[task])
            quotas[task] += take
            remaining += want - take
        open_tasks = [task for task in open_tasks if quotas[task] < len(by_task[task])]
    chosen = []
    for task in sorted(by_task):
        chosen += rng.sample(sorted(by_task[task], key=_entry_key), quotas[task])
    return sorted(chosen, key=_entry_key)


def discoverybench_examples(path, tasks, per_task=None, sample=None):
    benchmark = discovery_bench_hypothesis(path=path, num_samples=sys.maxsize)
    root = Path(benchmark.data_path)
    originals = sorted((e for e in benchmark.examples if e["answer"] is True and e["task"] in tasks), key=_entry_key)
    selected = []
    seen = set()
    taken = {}
    for entry in originals:
        key = (entry["task"], entry["prompt"])
        if key in seen or (per_task is not None and taken.get(entry["task"], 0) >= per_task):
            continue
        seen.add(key)
        taken[entry["task"]] = taken.get(entry["task"], 0) + 1
        selected.append(entry)
    if sample is not None:
        selected = stratified_sample(selected, int(sample["originals"]), sample["seed"])
    examples = []
    for entry in selected:
        task_path = root / entry["task"]
        metadata_path = task_path / f"metadata_{entry['metadataid']}.json"
        factory = functools.partial(discovery_factory, str(task_path), str(metadata_path))
        for permuted in (False, True):
            examples.append(
                {
                    "task": entry["task"],
                    "domain": entry["domain"],
                    "hypothesis": entry["prompt"],
                    "hypothesis_id": None,
                    "answer": not permuted,
                    "permuted": permuted,
                    "factory": factory,
                    "datasets": [entry["task"]],
                }
            )
    return examples


def gender_examples(hypotheses, data_dir, permutable, task=GENDER_TASK, domain=GENDER_DOMAIN):
    factory = functools.partial(custom_factory, str(data_dir), permutable)
    examples = []
    for hypothesis in hypotheses:
        base = {
            "task": task,
            "domain": domain,
            "hypothesis": hypothesis["text"],
            "hypothesis_id": hypothesis.get("id"),
            "factory": factory,
            "datasets": sorted(permutable),
        }
        examples.append({**base, "answer": False, "permuted": True})
        examples.append({**base, "answer": None, "permuted": False, "replicates": 1})
    return examples
