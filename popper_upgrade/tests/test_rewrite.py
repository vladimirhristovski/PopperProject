import json

from popper_upgrade.serving import rewrite as rewrite_module
from popper_upgrade.serving.rewrite import CHAT_PATH, request_rewriter
from popper_upgrade.serving.usage import identity_request


def _send(model, payload, path=CHAT_PATH):
    body = json.dumps(payload).encode()
    return body, request_rewriter(model)(path, body)


def test_models_without_settings_use_the_identity_rewriter():
    assert request_rewriter({"name": "m"}) is identity_request


def test_prefix_is_prepended_to_an_existing_system_message():
    payload = {"messages": [{"role": "system", "content": "POPPER rules"}, {"role": "user", "content": "hi"}]}

    _, sent = _send({"system_prefix": "PREFIX"}, payload)

    messages = json.loads(sent)["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "PREFIX\n\nPOPPER rules"


def test_prefix_becomes_the_system_message_when_there_is_none():
    _, sent = _send({"system_prefix": "PREFIX"}, {"messages": [{"role": "user", "content": "hi"}]})

    assert json.loads(sent)["messages"][0] == {"role": "system", "content": "PREFIX"}


def test_prefix_handles_list_content():
    payload = {"messages": [{"role": "system", "content": [{"type": "text", "text": "rules"}]}]}

    _, sent = _send({"system_prefix": "PREFIX"}, payload)

    assert json.loads(sent)["messages"][0]["content"] == [
        {"type": "text", "text": "PREFIX"},
        {"type": "text", "text": "rules"},
    ]


def test_drop_rename_and_set():
    overrides = {"drop": ["n"], "rename": {"max_tokens": "max_completion_tokens"}, "set": {"model": "real-id"}}

    _, sent = _send({"request_overrides": overrides}, {"model": "alias", "n": 1, "max_tokens": 9, "messages": []})

    assert json.loads(sent) == {"model": "real-id", "max_completion_tokens": 9, "messages": []}


def test_request_already_lacking_a_dropped_key_passes_unchanged():
    payload = {"messages": [{"role": "user", "content": "x"}]}

    body, sent = _send({"request_overrides": {"drop": ["temperature"]}}, payload)

    assert sent == body


def test_other_paths_and_bad_bodies_pass_unchanged():
    model = {"system_prefix": "PREFIX", "request_overrides": {"drop": ["a"]}}
    rewrite = request_rewriter(model)

    assert rewrite("/v1/models", b'{"a": 1}') == b'{"a": 1}'
    assert rewrite(CHAT_PATH, b"not json") == b"not json"


def test_inner_rewriter_runs_first():
    def inner(path, body):
        data = json.loads(body)
        data.pop("tools", None)
        return json.dumps(data).encode()

    rewrite = request_rewriter({"request_overrides": {"set": {"x": 1}}}, inner)

    sent = rewrite(CHAT_PATH, json.dumps({"tools": [1], "messages": []}).encode())

    assert json.loads(sent) == {"messages": [], "x": 1}


def _capped_model(**extra):
    return {
        "served_model_id": "org/m",
        "max_model_len": 12288,
        "request_overrides": {"max_output_tokens": 4096, "context_margin": 256, **extra},
    }


def _no_tokenizer(monkeypatch):
    monkeypatch.setattr(rewrite_module, "_tokenizer", lambda model_id, revision=None: None)


def _sent_max(model, payload, key="max_tokens"):
    _, sent = _send(model, payload)
    return json.loads(sent)[key]


def test_output_cap_applies_when_the_prompt_is_short(monkeypatch):
    _no_tokenizer(monkeypatch)

    assert _sent_max(_capped_model(), {"messages": [{"role": "user", "content": "hi"}]}) == 4096


def test_output_cap_shrinks_to_the_remaining_context(monkeypatch):
    _no_tokenizer(monkeypatch)
    content = "x" * (9217 * 3)
    payload = {"max_tokens": 3072, "messages": [{"role": "user", "content": content}]}

    sent = _sent_max(_capped_model(), payload)

    expected = 12288 - (9217 + rewrite_module.MESSAGE_OVERHEAD_TOKENS) - 256
    assert sent == expected
    assert sent + 9217 + 256 <= 12288


def test_output_cap_keeps_a_smaller_client_request(monkeypatch):
    _no_tokenizer(monkeypatch)

    assert _sent_max(_capped_model(), {"max_tokens": 100, "messages": []}) == 100


def test_output_cap_counts_tool_definitions(monkeypatch):
    _no_tokenizer(monkeypatch)
    payload = {"messages": [], "tools": [{"name": "t" * 3000}]}

    assert _sent_max(_capped_model(), payload) < 12288 - 256 - 1000


def test_output_cap_never_drops_below_one_token(monkeypatch):
    _no_tokenizer(monkeypatch)
    payload = {"messages": [{"role": "user", "content": "x" * 100000}]}

    assert _sent_max(_capped_model(), payload) == 1


def test_output_cap_without_a_context_length_is_the_plain_cap():
    model = {"request_overrides": {"max_output_tokens": 4096}}

    assert _sent_max(model, {"max_tokens": 9000, "messages": [{"role": "user", "content": "x" * 400000}]}) == 4096


def test_output_cap_follows_the_renamed_key():
    model = {
        "request_overrides": {"max_output_tokens": 4096, "rename": {"max_tokens": "max_completion_tokens"}},
    }

    _, sent = _send(model, {"max_tokens": 9000, "messages": []})

    assert json.loads(sent) == {"max_completion_tokens": 4096, "messages": []}


def test_output_cap_uses_the_model_tokenizer_when_available(monkeypatch):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

    monkeypatch.setattr(rewrite_module, "_tokenizer", lambda model_id, revision=None: Tokenizer())
    payload = {"messages": [{"role": "user", "content": " ".join(["w"] * 9000)}]}

    assert _sent_max(_capped_model(), payload) == 12288 - 9000 - rewrite_module.MESSAGE_OVERHEAD_TOKENS - 256


def test_tokenizer_failure_falls_back_to_the_character_estimate(monkeypatch):
    class Broken:
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(rewrite_module, "_tokenizer", lambda model_id, revision=None: Broken())
    payload = {"messages": [{"role": "user", "content": "x" * 30}]}

    assert rewrite_module.estimate_prompt_tokens("m", payload) == 10 + rewrite_module.MESSAGE_OVERHEAD_TOKENS


def test_tokenizer_is_loaded_at_the_pinned_revision_only(monkeypatch):
    import transformers

    calls = []

    def from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return "tokenizer"

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    rewrite_module._tokenizer.cache_clear()

    assert rewrite_module._tokenizer("org/pinned", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0") == "tokenizer"
    assert rewrite_module._tokenizer("org/plain") == "tokenizer"
    rewrite_module._tokenizer.cache_clear()
    assert calls == [
        (("org/pinned",), {"revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "local_files_only": True}),
        (("org/plain",), {"local_files_only": True}),
    ]


def test_output_cap_looks_up_the_tokenizer_with_the_models_revision(monkeypatch):
    seen = []
    monkeypatch.setattr(rewrite_module, "_tokenizer", lambda model_id, revision=None: seen.append((model_id, revision)))
    model = {**_capped_model(), "served_model_id": "org/m", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}

    request_rewriter(model)(CHAT_PATH, json.dumps({"messages": []}).encode())

    assert seen == [("org/m", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")]


def test_unavailable_tokenizer_is_cached_as_none(monkeypatch):
    rewrite_module._tokenizer.cache_clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert rewrite_module._tokenizer("no-such-org/no-such-model") is None
    rewrite_module._tokenizer.cache_clear()
