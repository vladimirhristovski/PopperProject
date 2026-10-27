import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from popper_upgrade.runner import PopperRunner
from popper_upgrade.serving.logs import is_server_failure
from popper_upgrade.skills.context import RunContext, describe_datasets
from popper_upgrade.skills.curation import PRECEDENT_COUNT, apply_overrides, curate_before_run

PINS_FILE = Path(__file__).parent / "requirements.txt"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunSpec:
    hypothesis: str
    hypothesis_id: str | None = None
    split: str = "none"
    arm: str = "baseline"
    model: dict = field(default_factory=dict)
    popper_kwargs: dict = field(default_factory=dict)
    register_kwargs: dict = field(default_factory=dict)
    configure_kwargs: dict = field(default_factory=dict)
    skills_enabled: bool = False
    reviewer_enabled: bool = False
    human_review: bool = False
    learn: bool = False
    environment: str = "laptop"
    launched_by: str | None = None
    experiment_id: str | None = None
    seed: int | None = None
    permuted: bool = False
    permute_seed: int | None = None
    library_snapshot_id: str | None = None
    data_loader_factory: object = None
    answer: bool | None = None
    task: str | None = None
    replicate: int | None = None
    datasets: list | None = None


def git_sha():
    if os.environ.get("POPPER_GIT_SHA"):
        return os.environ["POPPER_GIT_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def popper_version():
    try:
        return importlib.metadata.version("popper_agent")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def pins_hash():
    return hashlib.sha256(PINS_FILE.read_bytes()).hexdigest()


class RunSession:
    def __init__(
        self,
        spec,
        source,
        sink,
        curator,
        reviewer,
        extractor,
        trajectory_dir,
        overrides_provider=None,
        prompt_version_resolver=None,
        usage_recorder=None,
        raise_on_error=False,
        usage=None,
        on_start=None,
    ):
        if spec.skills_enabled and curator is None:
            raise ValueError("skills_enabled requires a curator")
        self._spec = spec
        self._source = source
        self._sink = sink
        self._curator = curator
        self._reviewer = reviewer
        self._extractor = extractor
        self._trajectory_dir = Path(trajectory_dir)
        self._overrides_provider = overrides_provider
        self._prompt_version_resolver = prompt_version_resolver
        self._usage_recorder = usage_recorder
        self._raise_on_error = raise_on_error
        self._usage = usage
        self._on_start = on_start
        self.runner = None

    def _config(self):
        spec = self._spec
        configure = spec.configure_kwargs
        data_path = spec.register_kwargs.get("data_path")
        datasets = spec.datasets if spec.datasets is not None else [Path(data_path).name] if data_path else []
        config = {
            "environment": spec.environment,
            "launched_by": spec.launched_by,
            "experiment_id": spec.experiment_id,
            "hypothesis_id": spec.hypothesis_id,
            "hypothesis": spec.hypothesis,
            "arm": spec.arm,
            "skills_enabled": spec.skills_enabled,
            "reviewer_enabled": spec.reviewer_enabled,
            "human_review": spec.human_review,
            "model_id": spec.model.get("id"),
            "model_name": spec.model.get("name", spec.popper_kwargs.get("llm")),
            "temperature": spec.model.get("temperature"),
            "seed": spec.seed,
            "alpha": configure.get("alpha", 0.1),
            "aggregate_test": configure.get("aggregate_test", "E-value"),
            "max_num_of_tests": configure.get("max_num_of_tests", 5),
            "max_retry": configure.get("max_retry", 5),
            "time_limit": configure.get("time_limit", 2),
            "domain": spec.popper_kwargs.get("domain", ""),
            "datasets": datasets,
            "permuted": spec.permuted,
            "permute_seed": spec.permute_seed,
            "library_snapshot_id": spec.library_snapshot_id,
            "answer": spec.answer,
            "task": spec.task,
            "replicate": spec.replicate,
            "popper_version": popper_version(),
            "git_sha": git_sha(),
            "pins_hash": pins_hash(),
        }
        return {key: value for key, value in config.items() if value is not None}

    def _build_runner(self):
        spec = self._spec
        runner = PopperRunner(
            skills_enabled=spec.skills_enabled,
            skill_source=self._source,
            allowed_skill_ids=set(),
            reviewer_enabled=spec.reviewer_enabled,
            reviewer=self._reviewer,
            data_loader_factory=spec.data_loader_factory,
            **spec.popper_kwargs,
        )
        if spec.register_kwargs:
            runner.register_data(**spec.register_kwargs)
        runner.configure(**spec.configure_kwargs)
        return runner

    def _tokens(self):
        if self._usage is None:
            return {}
        tokens_in, tokens_out = self._usage.snapshot()
        cost = self._usage.cost_usd() if hasattr(self._usage, "cost_usd") else None
        return {
            key: value
            for key, value in (("tokens_in", tokens_in), ("tokens_out", tokens_out), ("cost_usd", cost))
            if value is not None
        }

    def _prompt_version(self, role, text):
        if self._prompt_version_resolver is None:
            return None
        return self._prompt_version_resolver(role, text)

    def _curate(self, run_id, runner):
        spec = self._spec
        runner.prepare()
        context = RunContext(
            [spec.hypothesis], spec.popper_kwargs.get("domain", ""), describe_datasets(runner.data_tables())
        )
        result = curate_before_run(self._source, self._curator, context)
        if spec.human_review and self._overrides_provider is not None:
            result = apply_overrides(result, self._overrides_provider(result))
        self._record_curation(run_id, result)
        runner.allowed_skill_ids = set(result.approved_ids)

    def _record_curation(self, run_id, result):
        prompt_version_id = self._prompt_version("curator", self._curator.prompt_text)
        rows = []
        decisions = []
        for decision in result.decisions:
            role = "curator_approved" if decision.approved else "curator_rejected"
            rows.append({"skill_id": decision.skill_id, "role": role, "round": None, "similarity": None})
            decisions.append(
                {
                    "skill_id": decision.skill_id,
                    "run_id": run_id,
                    "trigger": "pre_run",
                    "decider": "curator",
                    "verdict": "approve" if decision.approved else "reject",
                    "reason": decision.reason,
                    "model": self._curator.model_name,
                    "prompt_version_id": prompt_version_id,
                }
            )
        for skill_id, approve in result.overrides.items():
            role = "human_override_approve" if approve else "human_override_reject"
            rows.append({"skill_id": skill_id, "role": role, "round": None, "similarity": None})
            decisions.append(
                {
                    "skill_id": skill_id,
                    "run_id": run_id,
                    "trigger": "pre_run",
                    "decider": "human",
                    "user_id": self._spec.launched_by,
                    "verdict": "approve" if approve else "reject",
                    "reason": "human override",
                }
            )
        self._sink.add_run_skills(run_id, rows)
        for decision in decisions:
            self._sink.add_skill_decision(decision)

    def _record_execution(self, run_id, runner):
        if runner.skill_agent is not None and runner.skill_agent.injection_log:
            self._sink.add_run_skills(
                run_id,
                [
                    {
                        "skill_id": entry["skill_id"],
                        "role": "injected",
                        "round": entry["round"],
                        "similarity": entry["similarity"],
                    }
                    for entry in runner.skill_agent.injection_log
                ],
            )
        if runner.review_gate is not None and runner.review_gate.reviews:
            self._sink.add_design_reviews(run_id, runner.review_gate.reviews)

    def _may_learn(self):
        spec = self._spec
        if self._extractor is None or self._curator is None:
            return False
        return spec.split == "train" or (spec.split == "none" and spec.learn)

    def _precedents(self, text, exclude_id=None):
        return [
            example
            for example in self._source.similar_decisions(text, PRECEDENT_COUNT)
            if example.skill.id != exclude_id
        ]

    def _learn(self, run_id, runner, traj):
        spec = self._spec
        rejected = [example for example in self._precedents(spec.hypothesis) if example.verdict == "reject"]
        candidates = self._extractor.extract(
            traj.tracked_tests, spec.hypothesis, spec.popper_kwargs.get("domain", ""), rejected, run_id
        )
        prompt_version_id = self._prompt_version("curator", self._curator.prompt_text)
        for candidate in candidates:
            if self._sink.add_candidate_skill(candidate) is None:
                continue
            try:
                decision = self._curator.review_candidate(
                    candidate, self._source.list_skills(True), self._precedents(candidate.applicability, candidate.id)
                )
            except Exception:
                continue
            self._sink.add_skill_decision(
                {
                    "skill_id": candidate.id,
                    "trigger": "entry",
                    "decider": "curator",
                    "verdict": "approve" if decision.approved else "reject",
                    "reason": decision.reason,
                    "model": self._curator.model_name,
                    "prompt_version_id": prompt_version_id,
                }
            )
        if spec.split == "train" and self._usage_recorder is not None and runner.skill_agent is not None:
            injected = {entry["skill_id"] for entry in runner.skill_agent.injection_log}
            for skill_id in sorted(injected):
                self._usage_recorder(skill_id, bool(traj.res))

    def _error_text(self, failure):
        text = repr(failure)
        if self._usage is None or not is_server_failure(text) or not hasattr(self._usage, "upstream_error"):
            return text
        detail = self._usage.upstream_error()
        return f"{text} | vLLM log: {detail}" if detail else text

    def _control_flags(self):
        return {"control_degenerate": True} if getattr(self._source, "degenerate", False) is True else {}

    def _write_trajectory(self, run_id, traj):
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = self._trajectory_dir / f"{run_id}.json"
        path.write_text(json.dumps(traj.to_dict(), default=str), encoding="utf-8")
        return str(path)

    def run(self):
        run_id = self._sink.create_run(self._config())
        if self._usage is not None:
            self._usage.reset()
        runner = self._build_runner()
        self.runner = runner
        started = time.monotonic()
        traj = None
        failure = None
        try:
            if self._on_start is not None:
                self._on_start(self)
            if self._spec.skills_enabled:
                self._curate(run_id, runner)
            traj = runner.validate(self._spec.hypothesis)
        except BaseException as e:
            failure = e
        self._record_execution(run_id, runner)
        if failure is not None:
            self._sink.finish_run(
                run_id,
                {
                    "status": "error",
                    "error": self._error_text(failure),
                    "wall_time_s": time.monotonic() - started,
                    **self._tokens(),
                    **self._control_flags(),
                },
            )
            if self._raise_on_error or not isinstance(failure, Exception):
                raise failure
            return run_id
        result = {
            "status": "completed",
            "res": bool(traj.res),
            "res_stat": None if traj.res_stat is None else float(traj.res_stat),
            "n_tests": len(traj.tracked_stat),
            "parsed_result": traj.parsed_result,
            "trajectory_path": self._write_trajectory(run_id, traj),
        }
        result.update(self._control_flags())
        if runner.review_gate is not None:
            result["reviewer_rejections"] = len(runner.review_gate.rejected)
            result["reviewer_overrides"] = runner.review_gate.overrides
        if self._may_learn():
            try:
                self._learn(run_id, runner, traj)
            except Exception as e:
                result["error"] = f"learning failed: {e!r}"
        result["wall_time_s"] = time.monotonic() - started
        result.update(self._tokens())
        self._sink.finish_run(run_id, result)
        return run_id
