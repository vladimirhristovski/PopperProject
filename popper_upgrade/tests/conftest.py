import pytest
from popper.agent import OutputSpecification

from popper_upgrade.runner import PopperRunner
from popper_upgrade.skills.schema import Skill


class _ApproveAllReviewer:
    def __init__(self):
        self.calls = []

    def review(self, main_hypothesis, proposal, prior_designs):
        self.calls.append((main_hypothesis, proposal, list(prior_designs)))
        return True, "ok"


class _OneSkillStore:
    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        return [
            (
                Skill(
                    id="s1",
                    applicability="a",
                    design_template="d",
                    preconditions="",
                    domain="social science",
                    source_hypothesis="h",
                ),
                0.9,
            )
        ]


class _Relevance:
    def dict(self):
        return {"relevance_score": 1.0, "relevance_reasoning": "ok"}


class _RelevanceChecker:
    def invoke(self, *args, **kwargs):
        return _Relevance()


class _CodingAgent:
    def go(self, proposal, log=None):
        return {"status": "ok", "p_val": 0.001}


class _OutputParser:
    def invoke(self, text):
        return OutputSpecification(
            main_hypothesis="H",
            falsification_test_result="pass",
            reasoning="r",
            conclusion=True,
            rationale="r",
        )


@pytest.fixture
def real_popper_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    data_dir = tmp_path / "data"
    (data_dir / "bio_database").mkdir(parents=True)
    rows = "\n".join(f"{i},{i * 2}" for i in range(10))
    (data_dir / "tiny.csv").write_text(f"a,b\n{rows}\n")

    original = PopperRunner._apply_hooks

    def hooked(self):
        agent = self._popper.agent
        agent.test_proposal_agent.go = lambda main_hypothesis, test_results=None, log=None: "Fixed proposal"
        original(self)
        agent.proposal_relevance_checker = _RelevanceChecker()
        agent.test_coding_agent = _CodingAgent()
        agent.output_parser = _OutputParser()
        agent.summarize = lambda: {"messages": [("assistant", "summary")]}

    monkeypatch.setattr(PopperRunner, "_apply_hooks", hooked)

    def build(skills_enabled, reviewer_enabled):
        runner = PopperRunner(
            llm="claude-3-5-sonnet-20241022",
            domain="social science",
            plot_agent_architecture=False,
            skills_enabled=skills_enabled,
            skill_source=_OneSkillStore() if skills_enabled else None,
            reviewer_enabled=reviewer_enabled,
            reviewer=_ApproveAllReviewer() if reviewer_enabled else None,
        )
        runner.register_data(str(data_dir), loader_type="custom")
        runner.configure(
            alpha=0.1,
            max_num_of_tests=3,
            max_retry=1,
            time_limit=1,
            aggregate_test="E-value",
            relevance_checker=True,
            use_react_agent=True,
        )
        return runner

    return build


@pytest.fixture
def job_env(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import popper_upgrade.models as models_module
    import popper_upgrade.run as run_module
    import popper_upgrade.runner as runner_module
    from cluster import common
    from popper_upgrade.tests.fakes import FakeEmbedder, FakePopper, patch_agent_classes, write_job

    class Server:
        def __init__(self, model, port, host="localhost", proxy_port=None):
            pass

        def start(self):
            return True

        def start_proxy(self):
            pass

        def stop_proxy(self):
            pass

        def stop(self):
            pass

    class Proxy:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class Popper(FakePopper):
        validations = []
        on_validate = None

        def validate(self, hypothesis):
            type(self).validations.append(hypothesis)
            if type(self).on_validate is not None:
                type(self).on_validate(len(type(self).validations))
            return super().validate(hypothesis)

    config, popper_common = common.load_baseline()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bbq.csv").write_text("context,question\nc1,q1\nc2,q2\nc3,q3\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(popper_common, "check_data", lambda: [])
    monkeypatch.setattr(common, "Embedder", FakeEmbedder)
    monkeypatch.setattr(models_module, "VllmServer", Server)
    monkeypatch.setattr(models_module, "UsageProxy", Proxy)
    monkeypatch.setattr(runner_module, "Popper", Popper)
    patch_agent_classes(monkeypatch, run_module)
    monkeypatch.setenv("POPPER_GIT_SHA", "placeholder")
    monkeypatch.delenv("POPPER_GIT_SHA")
    for name in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER", "HF_TOKEN", "HF_USERNAME", "POPPER_RESULTS_REPO"):
        monkeypatch.delenv(name, raising=False)
    job = write_job(tmp_path / "jobs" / "demo")
    return SimpleNamespace(
        job=job, job_dir=job["dir"], workdir=tmp_path / "work", popper=Popper, tmp_path=tmp_path
    )
