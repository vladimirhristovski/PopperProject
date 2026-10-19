import gradio as gr

from popper_upgrade.app.handlers import curation as handlers
from popper_upgrade.app.tabs.common import current_user, frame, guarded

COLUMNS = ("skill_id", "applicability", "approved", "reason")


def build(ctx):
    model = gr.Dropdown([], label="Model", elem_id="curation_model")
    hypothesis_set = gr.Dropdown([], label="Hypothesis set", elem_id="curation_set")
    hypotheses = gr.CheckboxGroup([], label="Hypotheses", elem_id="curation_hypotheses")
    datasets = gr.CheckboxGroup([], label="Datasets", elem_id="curation_datasets")
    run_button = gr.Button("Run curator")
    token = gr.Textbox(label="Curation token", interactive=False, elem_id="curation_token")
    table = gr.Dataframe(headers=list(COLUMNS), interactive=False, elem_id="curation_table")
    decisions_state = gr.State([])
    flips = gr.CheckboxGroup([], label="Flip the curator's decision for", elem_id="curation_flips")
    apply_button = gr.Button("Apply overrides")
    applied = gr.Markdown("", elem_id="curation_applied")

    def load_hypotheses(name):
        texts = handlers.set_hypotheses(ctx, name) if name else []
        return gr.update(choices=texts, value=texts)

    def refresh_choices(request: gr.Request):
        current_user(ctx, request)
        choices = handlers.picker_choices(ctx)
        sets = choices["hypothesis_sets"]
        return (
            gr.update(choices=choices["models"], value=choices["models"][0] if choices["models"] else None),
            gr.update(choices=sets, value=sets[0] if sets else None),
            load_hypotheses(sets[0] if sets else None),
            gr.update(choices=choices["datasets"], value=choices["datasets"]),
        )

    hypothesis_set.change(guarded(load_hypotheses), hypothesis_set, hypotheses, api_name=False)

    def preview_curation(model, hypotheses, datasets, request: gr.Request):
        user = current_user(ctx, request)
        result = handlers.preview_curation(ctx, user, model, hypotheses, datasets)
        decisions = result["decisions"]
        labels = [
            (f"{d['applicability'][:80]} ({'approved' if d['approved'] else 'rejected'})", d["skill_id"])
            for d in decisions
        ]
        return result["token"], frame(decisions, COLUMNS), decisions, gr.update(choices=labels, value=[]), ""

    run_button.click(
        guarded(preview_curation, 5),
        [model, hypotheses, datasets],
        [token, table, decisions_state, flips, applied],
        api_name="preview_curation",
    )

    def apply_overrides(token, flipped, decisions):
        overrides = {d["skill_id"]: not d["approved"] for d in decisions if d["skill_id"] in set(flipped)}
        handlers.apply_overrides(token, overrides)
        return f"Overrides applied: {len(overrides)}"

    apply_button.click(guarded(apply_overrides), [token, flips, decisions_state], applied, api_name="apply_overrides")
    return {"refresh": (refresh_choices, [model, hypothesis_set, hypotheses, datasets]), "token": token}
