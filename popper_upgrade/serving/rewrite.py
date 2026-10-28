import functools
import json
import math

from popper_upgrade.serving.usage import identity_request

CHAT_PATH = "/v1/chat/completions"
CHARS_PER_TOKEN = 3
MESSAGE_OVERHEAD_TOKENS = 8
DEFAULT_MARGIN = 256


@functools.cache
def _tokenizer(model_id, revision=None):
    try:
        from transformers import AutoTokenizer

        if revision:
            return AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return None


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _prompt_text(data):
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    parts = [_text_of(m.get("content")) for m in messages if isinstance(m, dict)]
    if data.get("tools"):
        parts.append(json.dumps(data["tools"]))
    return parts, len(messages)


def estimate_prompt_tokens(model_id, data, revision=None):
    parts, count = _prompt_text(data)
    tokenizer = _tokenizer(model_id, revision) if model_id else None
    if tokenizer is not None:
        try:
            return sum(len(tokenizer.encode(part, add_special_tokens=False)) for part in parts) + (
                MESSAGE_OVERHEAD_TOKENS * count
            )
        except Exception:
            pass
    return math.ceil(sum(len(part) for part in parts) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS * count


def _prefixed(messages, prefix):
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content")
        if isinstance(content, str):
            first["content"] = f"{prefix}\n\n{content}"
        elif isinstance(content, list):
            first["content"] = [{"type": "text", "text": prefix}, *content]
        else:
            first["content"] = prefix
        return [first, *messages[1:]]
    return [{"role": "system", "content": prefix}, *messages]


def _overridden(data, overrides):
    for key in overrides.get("drop", []):
        data.pop(key, None)
    for old, new in overrides.get("rename", {}).items():
        if old in data:
            value = data.pop(old)
            data.setdefault(new, value)
    data.update(overrides.get("set", {}))
    return data


def _capped(data, model, overrides):
    cap = overrides.get("max_output_tokens")
    if not cap:
        return data
    key = overrides.get("rename", {}).get("max_tokens", "max_tokens")
    limit = cap
    context = model.get("max_model_len")
    if context:
        margin = overrides.get("context_margin", DEFAULT_MARGIN)
        room = context - estimate_prompt_tokens(model.get("served_model_id"), data, model.get("revision")) - margin
        limit = min(limit, max(room, 1))
    requested = data.get(key)
    if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
        limit = min(limit, requested)
    data[key] = limit
    return data


def request_rewriter(model, inner=identity_request):
    prefix = model.get("system_prefix")
    overrides = model.get("request_overrides") or {}
    if not prefix and not overrides:
        return inner

    def rewrite(path, body):
        body = inner(path, body)
        if path != CHAT_PATH:
            return body
        try:
            data = json.loads(body)
            before = json.loads(body)
            if prefix and isinstance(data.get("messages"), list):
                data["messages"] = _prefixed(data["messages"], prefix)
            data = _capped(_overridden(data, overrides), model, overrides)
            return body if data == before else json.dumps(data).encode()
        except Exception as e:
            print(f"  [proxy] failed to apply model request settings: {e}")
            return body

    return rewrite
