import gradio as gr

from popper_upgrade.app.handlers import curation as curation_handlers
from popper_upgrade.app.tabs.common import current_user, guarded, log_chatbots, stream_updates

ARMS = ["baseline", "skills", "skills_reviewer"]


def build(ctx, launcher, token):
    model = gr.Dropdown([], label="Model", elem_id="launch_model")
    hypothesis_set = gr.Dropdown([], label="Hypothesis set", elem_id="launch_set")
    indices = gr.CheckboxGroup([], label="Hypotheses", type="index", elem_id="launch_hypotheses")
    arm = gr.Radio(ARMS, value="baseline", label="Arm")
    with gr.Row():
        human_review = gr.Checkbox(label="Human review (apply curation overrides)")
        learn = gr.Checkbox(label="Learn skills from this run")
    launch_token = gr.Textbox(label="Curation token", elem_id="launch_token")
    with gr.Row():
        start_button = gr.Button("Start run", variant="primary")
        stop_button = gr.Button("Stop after current hypothesis")
    status = gr.Markdown("Idle", elem_id="launch_status")
    chatbots = log_chatbots()
    timer = gr.Timer(1)

    def load_hypotheses(name):
        texts = curation_handlers.set_hypotheses(ctx, name) if name else []
        return gr.update(choices=texts, value=[])

    def refresh_choices(request: gr.Request):
        current_user(ctx, request)
        choices = curation_handlers.picker_choices(ctx)
        sets = choices["hypothesis_sets"]
        return (
            gr.update(choices=choices["models"], value=choices["models"][0] if choices["models"] else None),
            gr.update(choices=sets, value=sets[0] if sets else None),
            load_hypotheses(sets[0] if sets else None),
        )

    hypothesis_set.change(guarded(load_hypotheses), hypothesis_set, indices, api_name=False)
    token.change(lambda value: value, token, launch_token, api_name=False)

    def launch_start(model, hypothesis_set, indices, arm, human_review, launch_token, learn, request: gr.Request):
        user = current_user(ctx, request)
        launcher.start(
            user, model, hypothesis_set, indices or [], arm, human_review, (launch_token or "").strip() or None, learn
        )
        gr.Info("Run started")
        return "Started"

    def launch_stop():
        launcher.stop()
        return "Stopping after the current hypothesis"

    def launch_status():
        state = launcher.status()
        if state["run_id"] is None:
            text = "Finished" if state["done"] else "Starting"
        else:
            text = f"{'Finished' if state['done'] else 'Running'}: {state['hypothesis']} (run {state['run_id']})"
        if state["error"]:
            text += f"\n\nError: {state['error']}"
        return [text, *stream_updates(state["log"])]

    start_button.click(
        guarded(launch_start),
        [model, hypothesis_set, indices, arm, human_review, launch_token, learn],
        status,
        api_name="launch_start",
    )
    stop_button.click(guarded(launch_stop), None, status, api_name="launch_stop")
    timer.tick(launch_status, None, [status, *chatbots.values()], api_name="launch_status", show_progress="hidden")
    return {"refresh": (refresh_choices, [model, hypothesis_set, indices])}
