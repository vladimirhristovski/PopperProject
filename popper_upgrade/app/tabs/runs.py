import gradio as gr

from popper_upgrade.app.handlers import runs as handlers
from popper_upgrade.app.tabs.common import current_user, frame, guarded

COLUMNS = (
    "id",
    "created_at",
    "hypothesis",
    "arm",
    "model_name",
    "environment",
    "status",
    "res",
    "res_stat",
    "n_tests",
)


def build(ctx, run_id):
    with gr.Row():
        model = gr.Textbox(label="Model")
        arm = gr.Dropdown(["", "baseline", "skills", "skills_reviewer"], value="", label="Arm")
        status = gr.Dropdown(["", "running", "completed", "error"], value="", label="Status")
        environment = gr.Dropdown(["", "laptop", "cluster"], value="", label="Environment")
        experiment = gr.Textbox(label="Experiment")
        refresh = gr.Button("Refresh runs")
    table = gr.Dataframe(headers=list(COLUMNS), interactive=False, elem_id="runs_table")

    def list_runs(model, arm, status, environment, experiment, request: gr.Request):
        user = current_user(ctx, request)
        filters = {
            "model": model,
            "arm": arm,
            "status": status,
            "environment": environment,
            "experiment": experiment,
        }
        return frame(handlers.list_runs(ctx, user, filters), COLUMNS)

    inputs = [model, arm, status, environment, experiment]
    refresh.click(guarded(list_runs), inputs, table, api_name="list_runs")

    def pick_run(evt: gr.SelectData):
        return evt.row_value[0]

    table.select(pick_run, None, run_id, api_name=False)
    return {"load": (list_runs, inputs, [table])}
