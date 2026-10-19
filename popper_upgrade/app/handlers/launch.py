import threading

from popper_upgrade.app.handlers.curation import overrides_for
from popper_upgrade.db.audit import log_event
from popper_upgrade.db.connection import connect
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.models import check_environment, serving_for
from popper_upgrade.run import run_hypotheses

LOG_KEYS = ("designer", "executor", "relevance_checker", "summarizer", "sequential_testing")


class _TrackingSink:
    def __init__(self, sink, manager):
        self._sink = sink
        self._manager = manager

    def create_run(self, config):
        run_id = self._sink.create_run(config)
        self._manager._track(run_id, config.get("hypothesis"))
        return run_id

    def __getattr__(self, name):
        return getattr(self._sink, name)


def _format_review(review):
    verdict = "approved" if review["approved"] else "rejected"
    overridden = " (overridden)" if review.get("overridden") else ""
    return f"round {review['round']} attempt {review['attempt']}: {verdict}{overridden} - {review['reason']}"


class LaunchManager:
    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._session = None
        self._run_id = None
        self._hypothesis = None
        self._error = None

    def _active(self):
        return self._thread is not None and self._thread.is_alive()

    def _track(self, run_id, hypothesis):
        self._run_id = run_id
        self._hypothesis = hypothesis

    def _on_session(self, session):
        self._session = session

    def _work(self, model, hypotheses, arm, user_id, human_review, learn, overrides_provider):
        conn = connect(self._ctx.dsn)
        conn.autocommit = True
        try:
            skills = SkillRepository(conn, self._ctx.embedder)
            sink = _TrackingSink(RunRepository(conn, skills), self)
            with serving_for(model) as counter:
                run_hypotheses(
                    model,
                    hypotheses,
                    arm,
                    skills,
                    sink,
                    "laptop",
                    human_review=human_review,
                    learn=learn,
                    launched_by=user_id,
                    overrides_provider=overrides_provider,
                    on_session_start=self._on_session,
                    usage=counter,
                    trajectory_dir=self._ctx.trajectory_dir,
                    stop_requested=self._stop.is_set,
                )
        except Exception as e:
            self._error = repr(e)
        finally:
            conn.close()

    def start(
        self, user, model_name, hypothesis_set, hypothesis_indices, arm, human_review, curation_token, learn=False
    ):
        with self._lock:
            if self._active():
                raise RuntimeError("a run is already in progress")
            with self._ctx.conn() as conn:
                registry = self._ctx.registry(conn)
                model = registry.model(model_name)
                check_environment(model, "laptop")
                available = registry.hypothesis_set(hypothesis_set)
                if not hypothesis_indices:
                    raise ValueError("select at least one hypothesis")
                hypotheses = [
                    {"id": available[i]["id"], "text": available[i]["text"], "split": available[i]["split"]}
                    for i in hypothesis_indices
                ]
                if curation_token:
                    overrides_for(curation_token)
                log_event(
                    conn,
                    user.id,
                    "run.launch",
                    "run",
                    hypothesis_set,
                    after={"model": model_name, "arm": arm, "indices": list(hypothesis_indices)},
                )
            provider = None
            if curation_token:
                human_review = True

                def provider(_result, token=curation_token):
                    return overrides_for(token)

            self._stop.clear()
            self._session = None
            self._run_id = None
            self._hypothesis = None
            self._error = None
            self._thread = threading.Thread(
                target=self._work,
                args=(model, hypotheses, arm, user.id, human_review, learn, provider),
                daemon=True,
            )
            self._thread.start()
        return "started"

    def stop(self):
        self._stop.set()

    def status(self):
        log = {key: [] for key in LOG_KEYS}
        log["reviewer"] = []
        session = self._session
        runner = session.runner if session is not None else None
        agent = runner.agent if runner is not None else None
        if agent is not None:
            for key in LOG_KEYS:
                log[key] = list(agent.log.get(key, []))
        if runner is not None and runner.review_gate is not None:
            log["reviewer"] = [_format_review(review) for review in list(runner.review_gate.reviews)]
        return {
            "run_id": self._run_id,
            "hypothesis": self._hypothesis,
            "log": log,
            "done": not self._active(),
            "error": self._error,
        }
