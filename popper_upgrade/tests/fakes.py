import hashlib
import json
import threading
import urllib.request
import uuid
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd

from popper_upgrade.skills.curator import CRITERIA, CuratorDecision


class FakeEmbedder:
    def __init__(self, dim=384, model_name="fake-embedder"):
        self.dim = dim
        self.model_name = model_name

    def encode(self, text):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        vector = np.random.default_rng(seed).standard_normal(self.dim)
        return (vector / np.linalg.norm(vector)).tolist()


def write_snapshot(
    path,
    skills,
    embedder,
    decisions=(),
    embedding_model=None,
    snapshot_id="snap-1",
    models=(),
    hypothesis_sets=None,
    prompt_versions=(),
    datasets=(),
):
    entries = [{**asdict(skill), "embedding": embedder.encode(skill.applicability)} for skill in skills]
    payload = {
        "snapshot_id": snapshot_id,
        "created_at": "2026-10-18T00:00:00+00:00",
        "embedding_model": embedding_model or embedder.model_name,
        "skills": entries,
        "decisions": list(decisions),
        "prompt_versions": list(prompt_versions),
        "models": list(models),
        "hypothesis_sets": hypothesis_sets or {},
        "datasets": list(datasets),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_config(**overrides):
    config = {
        "environment": "laptop",
        "hypothesis": "H",
        "arm": "baseline",
        "skills_enabled": False,
        "reviewer_enabled": False,
        "human_review": False,
        "model_name": "claude-sonnet-4-5",
        "alpha": 0.1,
        "aggregate_test": "E-value",
        "max_num_of_tests": 3,
        "max_retry": 1,
        "time_limit": 10,
        "domain": "social science",
        "datasets": ["winobias"],
        "popper_version": "0.0.5",
        "git_sha": "abc",
        "pins_hash": "def",
    }
    config.update(overrides)
    return config


class FakeChain:
    def __init__(self, output):
        self._output = output
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        if callable(self._output):
            return self._output(inputs)
        return self._output


class _FakeModel:
    def __init__(self, chain):
        self._chain = chain

    def with_structured_output(self, schema):
        return self._chain


class _FakePromptTemplate:
    def __or__(self, other):
        return other


def patch_chain(monkeypatch, module, output):
    chain = FakeChain(output)
    monkeypatch.setattr(module, "get_llm", lambda llm, port=None, api_key="EMPTY": _FakeModel(chain))
    monkeypatch.setattr(
        module.ChatPromptTemplate, "from_messages", staticmethod(lambda messages: _FakePromptTemplate())
    )
    return chain


class FakeProposalAgent:
    def __init__(self):
        self.existing_tests = []
        self.failed_tests = []

    def go(self, main_hypothesis, test_results=None, log=None):
        return "proposal"

    def add_to_existing_tests(self, test):
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test):
        self.failed_tests.append(test)


class FakeAgent:
    def __init__(self):
        self.log = {"designer": [], "executor": []}
        self.tracked_tests = []
        self.tracked_stat = []
        self.res = True
        self.res_stat = 42.0
        self.test_proposal_agent = FakeProposalAgent()


class FakeLoader:
    def __init__(self):
        self.table_dict = {"df_winobias": pd.DataFrame({"gender": ["f", "m"], "score": [1.0, 2.0]})}


class FakePopper:
    failure = None

    def __init__(self, **kwargs):
        self.agent = FakeAgent()
        self.data_loader = FakeLoader()

    def register_data(self, data_path, loader_type="bio", **kwargs):
        pass

    def configure(self, **kwargs):
        pass

    def validate(self, hypothesis):
        if self.failure is not None:
            raise self.failure
        self.agent.test_proposal_agent.go(hypothesis, "No Implemented Falsification Test Yet.", {})
        self.agent.tracked_tests = ["design one"]
        self.agent.tracked_stat = [0.01]
        return {"parsed_result": {"conclusion": True}, "last_message": "done"}


class FakeCurator:
    model_name = "fake-curator"
    prompt_text = "curator prompt"

    def __init__(self, rejected=(), candidate_approved=True, candidate_error=None):
        self._rejected = set(rejected)
        self._candidate_approved = candidate_approved
        self._candidate_error = candidate_error
        self.run_calls = []
        self.candidate_calls = []

    def curate_for_run(self, skills, context, precedents):
        self.run_calls.append((list(skills), context, precedents))
        decisions = []
        for skill in skills:
            approved = skill.id not in self._rejected
            decisions.append(CuratorDecision(skill.id, approved, "run reason", dict.fromkeys(CRITERIA, approved)))
        return decisions

    def review_candidate(self, skill, approved_library, precedents):
        self.candidate_calls.append(skill)
        if self._candidate_error is not None:
            raise self._candidate_error
        return CuratorDecision(
            skill.id, self._candidate_approved, "entry reason", dict.fromkeys(CRITERIA, self._candidate_approved)
        )


class FakeExtractor:
    prompt_text = "extractor prompt"

    def __init__(self, skills=()):
        self._skills = list(skills)
        self.calls = []

    def extract(self, designs, hypothesis, domain, rejected_examples, source_run_id):
        self.calls.append((list(designs), hypothesis, domain, list(rejected_examples), source_run_id))
        return [
            replace(skill, id=str(uuid.uuid4()), source_run_id=source_run_id, approved=None) for skill in self._skills
        ]


class FakeSink:
    def __init__(self):
        self.config = None
        self.result = None
        self.run_skills = []
        self.design_reviews = []
        self.decisions = []
        self.candidates = []
        self.configs = []
        self.results = []
        self.experiments = []
        self.completed = set()

    def create_experiment(self, experiment):
        self.experiments.append(experiment)
        return experiment["id"]

    def completed_runs(self, experiment_id):
        return set(self.completed)

    def create_run(self, config):
        self.config = config
        self.configs.append(config)
        return f"run-{len(self.configs)}"

    def finish_run(self, run_id, result):
        self.result = result
        self.results.append((run_id, result))

    def add_run_skills(self, run_id, rows):
        self.run_skills.extend(rows)

    def add_design_reviews(self, run_id, reviews):
        self.design_reviews.extend(reviews)

    def add_skill_decision(self, decision):
        self.decisions.append(decision)
        return str(len(self.decisions))

    def add_candidate_skill(self, skill):
        self.candidates.append(skill)
        return skill.id


class FakeUpstream:
    def __init__(self, body):
        self.body = body
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.headers.append(dict(self.headers))
                outer.requests.append((self.path, self.rfile.read(length)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def post_chat(port, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def model_entry(name, backend, environment, port=None, **extra):
    return {
        "id": f"id-{name}",
        "name": name,
        "backend": backend,
        "served_model_id": name,
        "size_class": "small",
        "environment": environment,
        "port": port,
        "default_temperature": None,
        "base_url": None,
        "api_key_env": None,
        "strip_tools": True,
        **extra,
    }


OLLAMA_MODEL = model_entry("llama3.1:8b", "ollama", "laptop", 11435)
VLLM_MODEL = model_entry("casperhansen/llama-3.3-70b-instruct-awq", "vllm", "cluster", 8001)
HOSTED_MODEL = model_entry("claude-sonnet-4-5", "anthropic", "laptop")
OPENAI_MODEL = model_entry("gpt-4o", "openai", "laptop")
COMPATIBLE_MODEL = model_entry(
    "custom-endpoint-model",
    "openai_compatible",
    "laptop",
    base_url="http://127.0.0.1:9/v1",
    api_key_env="CUSTOM_ENDPOINT_KEY",
)
HYPOTHESES = [{"id": f"h{i}", "text": f"hypothesis {i}", "split": "none"} for i in range(5)]


class FakeReviewer:
    def review(self, main_hypothesis, proposal, prior_designs):
        return True, "ok"


def _recording_factory(created, maker):
    def build(llm, **kwargs):
        created.append((llm, kwargs))
        return maker()

    return build


def patch_agent_classes(monkeypatch, module):
    created = {"SkillCurator": [], "Reviewer": [], "SkillExtractor": []}
    makers = {"SkillCurator": FakeCurator, "Reviewer": FakeReviewer, "SkillExtractor": FakeExtractor}
    for name, maker in makers.items():
        monkeypatch.setattr(module, name, _recording_factory(created[name], maker))
    return created


JOB_DATASETS = [{"name": "bbq", "permutable_columns": ["question"]}]


def write_job(
    directory, name="demo", model=None, hypotheses=None, arms=("baseline",), replicates=2, train=None, **model_extra
):
    from popper_upgrade.eval.run_experiment import current_prompt_hashes

    model = {**(model or VLLM_MODEL), **model_extra}
    directory.mkdir(parents=True, exist_ok=True)
    write_snapshot(
        directory / "snapshot.json",
        [],
        FakeEmbedder(),
        models=[model],
        hypothesis_sets={"gender_v1": hypotheses if hypotheses is not None else HYPOTHESES[:2]},
        datasets=JOB_DATASETS,
    )
    spec = {
        "name": f"{name}_spec",
        "models": [model["name"]],
        "arms": list(arms),
        "source": {"kind": "gender_custom", "hypothesis_set": "gender_v1", "task": "gender_bias"},
        "train_tasks": [],
        "test_tasks": ["gender_bias"],
        "replicates": replicates,
        "permutation_seeds": list(range(5, 5 + replicates)),
        "prompt_versions": current_prompt_hashes(),
    }
    if train:
        spec["source"] = {**spec["source"], "train_task": "gender_bias_train", "train_hypotheses": list(train)}
        spec["train_tasks"] = ["gender_bias_train"]
    (directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    job = {
        "name": name,
        "model": model["name"],
        "spec": "spec.json",
        "snapshot": "snapshot.json",
        "created_at": "2026-10-19T00:00:00+00:00",
        "git_sha": "job-build-sha",
    }
    (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return {**job, "dir": str(directory)}
