import gradio as gr
import pandas as pd

from popper_upgrade.app.handlers import runs as handlers
from popper_upgrade.app.tabs.common import (
    LOG_STREAMS,
    chat_messages,
    current_user,
    guarded,
    jsonable,
    log_chatbots,
)

CHART_COLUMNS = ["test", "value", "series"]
THRESHOLD_STEPS = 20


def chart_frame(series, alpha):
    threshold = 1 / alpha
    span = max(len(series), 1) + 1
    rows = [
        {"test": round(span * step / THRESHOLD_STEPS, 3), "value": threshold, "series": "threshold"}
        for step in range(THRESHOLD_STEPS + 1)
    ]
    rows += [{"test": point["test"], "value": point["e_value"], "series": "e-value"} for point in series]
    return pd.DataFrame(rows, columns=CHART_COLUMNS)


def _stream_entries(detail):
    trajectory = detail["trajectory"]
    log = dict(trajectory["log"]) if trajectory else {}
    log["reviewer"] = [
        f"round {r['round']} attempt {r['attempt']}: {'approved' if r['approved'] else 'rejected'}"
        f"{' (overridden)' if r['overridden'] else ''} - {r['reason']}"
        for r in detail["design_reviews"]
    ]
    return log


def build(ctx, run_id):
    run_id.render()
    load_button = gr.Button("Load run")
    run_row = gr.JSON(label="Run", elem_id="run_row")
    with gr.Row():
        skills = gr.JSON(label="Skills by role")
        reviews = gr.JSON(label="Design reviews")
    chart = gr.ScatterPlot(
        x="test",
        y="value",
        color="series",
        title="E-value vs rejection threshold",
        elem_id="evalue_chart",
    )
    chatbots = log_chatbots()

    def run_detail(run_id, request: gr.Request):
        user = current_user(ctx, request)
        detail = handlers.run_detail(ctx, user, run_id)
        run = detail["run"]
        trajectory = detail["trajectory"]
        series = handlers.evalue_series(trajectory["tracked_stat"] if trajectory else [], run["alpha"])
        log = _stream_entries(detail)
        return (
            jsonable(run),
            jsonable(detail["skills"]),
            jsonable(detail["design_reviews"]),
            chart_frame(series, run["alpha"]),
            *[chat_messages(log.get(stream, [])) for stream in LOG_STREAMS],
        )

    outputs = [run_row, skills, reviews, chart, *chatbots.values()]
    load_button.click(guarded(run_detail, len(outputs)), run_id, outputs, api_name="run_detail")
    return {}
