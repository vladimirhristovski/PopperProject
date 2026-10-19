import functools
import json

import gradio as gr
import pandas as pd

from popper_upgrade.app.auth import user_by_name

LOG_STREAMS = ("designer", "executor", "relevance_checker", "summarizer", "sequential_testing", "reviewer")


def current_user(ctx, request):
    username = getattr(request, "username", None)
    if not username:
        raise PermissionError("login required")
    with ctx.conn() as conn:
        user = user_by_name(conn, username)
    if user is None:
        raise PermissionError("unknown user")
    return user


def guarded(fn, outputs=1):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            gr.Warning(str(e) or type(e).__name__)
            return tuple(gr.update() for _ in range(outputs)) if outputs > 1 else gr.update()

    return wrapper


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def frame(rows, columns):
    if not rows:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame([{column: row.get(column) for column in columns} for row in rows], columns=list(columns))


def chat_messages(entries):
    return [{"role": "assistant", "content": str(entry)} for entry in entries]


def log_chatbots():
    return {stream: gr.Chatbot(type="messages", label=stream, height=260) for stream in LOG_STREAMS}


def stream_updates(log):
    return [chat_messages(log.get(stream, [])) for stream in LOG_STREAMS]
