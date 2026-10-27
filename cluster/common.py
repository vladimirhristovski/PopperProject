import contextlib
import importlib
import os
import signal
import sys
from pathlib import Path

from popper_upgrade.db.bundle import BundleSink
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.learning import bundle_learning, open_library
from popper_upgrade.eval.run_experiment import run_experiment
from popper_upgrade.models import ServingError, check_environment, serving_for
from popper_upgrade.run import run_hypotheses
from popper_upgrade.skills.embedding import Embedder

BASELINE_DIR = Path(
    os.environ.get("POPPER_BASELINE_DIR", Path(__file__).resolve().parent.parent / "baseline_experiment")
)
ARMS = ("baseline", "skills", "skills_reviewer")
HYPOTHESIS_SET = "gender_v1"
LOCAL_HOST = "localhost"


class Interrupted(BaseException):
    pass


def load_baseline():
    if str(BASELINE_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINE_DIR))
    return importlib.import_module("config"), importlib.import_module("popper_common")


def baseline_configure_kwargs(config):
    return {
        "alpha": config.ALPHA,
        "max_num_of_tests": config.MAX_TESTS,
        "max_retry": config.MAX_RETRY,
        "time_limit": config.TIME_LIMIT,
        "aggregate_test": "E-value",
        "relevance_checker": True,
        "use_react_agent": True,
    }


class _Guard:
    def __init__(self):
        self.signum = None
        self.active = False

    def handle(self, signum, frame):
        self.signum = signum
        if self.active:
            self.active = False
            raise Interrupted(f"received signal {signum}")


class SetupError(Exception):
    pass


def _load_config():
    config, popper_common = load_baseline()
    missing = popper_common.check_data()
    if missing:
        raise SetupError(f"Missing data files: {missing}\nRun: python3 prepare_data.py")
    return config


def _prepare(model_name, snapshot_path):
    config = _load_config()
    snapshot = SnapshotSkillSource(snapshot_path, Embedder())
    model = next((m for m in snapshot.registry["models"] if m["name"] == model_name), None)
    if model is None:
        raise SetupError(f"model {model_name} is not in the snapshot registry")
    try:
        check_environment(model, "cluster")
    except ValueError as e:
        raise SetupError(str(e)) from e
    hypotheses = snapshot.registry["hypothesis_sets"].get(HYPOTHESIS_SET)
    if not hypotheses:
        raise SetupError(f"hypothesis set {HYPOTHESIS_SET} is not in the snapshot registry")
    return config, snapshot, model, hypotheses


@contextlib.contextmanager
def _signals():
    guard = _Guard()
    previous = {sig: signal.signal(sig, guard.handle) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield guard
    finally:
        guard.active = False
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _export_serving_environment(counter, model):
    os.environ["OPENAI_API_KEY"] = "vllm"
    os.environ["OPENAI_BASE_URL"] = f"http://{LOCAL_HOST}:{counter.port or model['port']}/v1"


def _exit_code(guard):
    return 128 + guard.signum if guard.signum is not None else 0


def run_model(model_name, snapshot_path, bundle_dir, arms=ARMS):
    try:
        config, snapshot, model, hypotheses = _prepare(model_name, snapshot_path)
    except SetupError as e:
        print(f"ERROR: {e}")
        return 1

    bundle_dir = Path(bundle_dir)
    sink = BundleSink(bundle_dir, snapshot)
    with _signals() as guard:
        try:
            guard.active = True
            with serving_for(model) as counter:
                _export_serving_environment(counter, model)
                try:
                    for arm in arms:
                        if guard.signum is not None:
                            break
                        run_hypotheses(
                            model,
                            hypotheses,
                            arm,
                            snapshot,
                            sink,
                            "cluster",
                            usage=counter,
                            trajectory_dir=bundle_dir / "trajectories",
                            stop_requested=lambda: guard.signum is not None,
                            configure_kwargs=baseline_configure_kwargs(config),
                            data_dir=config.DATA_DIR,
                        )
                finally:
                    guard.active = False
        except ServingError as e:
            print(f"ERROR: {e}")
            return 1
        except Interrupted:
            pass
    return _exit_code(guard)


def run_experiment_job(
    spec_path,
    model_name,
    snapshot_path,
    bundle_dir,
    tasks=None,
    replicates=None,
    per_task=None,
    budget=None,
    library_from=None,
    log_dir=None,
):
    try:
        config = _load_config()
        embedder = Embedder()
        snapshot = SnapshotSkillSource(snapshot_path, embedder)
    except SetupError as e:
        print(f"ERROR: {e}")
        return 1

    bundle_dir = Path(bundle_dir)
    frozen = None
    if library_from is not None:
        try:
            frozen = SnapshotSkillSource(library_from, embedder)
        except (OSError, ValueError, KeyError) as e:
            print(f"ERROR: {e}")
            return 1
        learning = None
        sink = BundleSink(bundle_dir, frozen)
    else:
        library = open_library(snapshot_path, bundle_dir, embedder)
        sink = BundleSink(bundle_dir, library)
        learning = bundle_learning(snapshot_path, bundle_dir, embedder, library, sink)
    with _signals() as guard:
        try:
            guard.active = True
            run_experiment(
                spec_path,
                model_name,
                snapshot,
                sink,
                "cluster",
                tasks=tasks,
                replicates=replicates,
                per_task=per_task,
                data_dir=config.DATA_DIR,
                trajectory_dir=bundle_dir / "trajectories",
                stop_requested=lambda: guard.signum is not None,
                on_serving=_export_serving_environment,
                learning=learning,
                budget=budget,
                library=frozen,
                log_dir=log_dir,
            )
        except (ValueError, KeyError, ServingError) as e:
            print(f"ERROR: {e}")
            return 1
        except Interrupted:
            pass
        finally:
            guard.active = False
    return _exit_code(guard)
