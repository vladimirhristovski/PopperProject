import json
import os

from popper_upgrade.serving.usage import RelayProxy, UsageCounter

DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
CHAT_PATH = "/v1/chat/completions"


def strip_tools(path, body):
    if path != CHAT_PATH:
        return body
    try:
        data = json.loads(body)
        data.pop("tools", None)
        data.pop("tool_choice", None)
        return json.dumps(data).encode()
    except Exception as e:
        print(f"  [proxy] failed to strip tools from request: {e}")
        return body


def decode_arguments(body):
    try:
        data = json.loads(body)
        changed = False
        for choice in data.get("choices", []):
            message = choice.get("message") or choice.get("delta") or {}
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                        changed = True
                    except Exception as e:
                        print(f"  [proxy] failed to decode tool_call arguments: {e}")
        return json.dumps(data).encode() if changed else body
    except Exception as e:
        print(f"  [proxy] failed to post-process response: {e}")
        return body


class OllamaProxy(RelayProxy):
    def __init__(self, listen_port=11435, upstream=None, counter=None):
        super().__init__(
            listen_port,
            upstream or os.environ.get("POPPER_OLLAMA_UPSTREAM", DEFAULT_UPSTREAM),
            counter or UsageCounter(),
            strip_tools,
            decode_arguments,
        )
