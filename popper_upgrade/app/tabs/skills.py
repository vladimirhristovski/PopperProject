import gradio as gr

from popper_upgrade.app.handlers import skills as handlers
from popper_upgrade.app.tabs.common import current_user, frame, guarded, jsonable

COLUMNS = ("id", "applicability", "domain", "status", "flagged", "times_used", "win_rate", "created_at")


def _badge(rows):
    pending = sum(1 for row in rows if row["status"] == "pending")
    flagged = sum(1 for row in rows if row["flagged"])
    return f"Pending: {pending} | Flagged: {flagged}"


def build(ctx):
    with gr.Row():
        status = gr.Dropdown(list(handlers.STATUSES), value="all", label="Status")
        domain = gr.Textbox(label="Domain")
        text = gr.Textbox(label="Text")
        refresh = gr.Button("Refresh skills")
    badge = gr.Markdown("Pending: 0 | Flagged: 0", elem_id="skills_badge")
    table = gr.Dataframe(headers=list(COLUMNS), interactive=False, elem_id="skills_table")

    def list_skills(status, domain, text, request: gr.Request):
        user = current_user(ctx, request)
        rows = handlers.list_skills(ctx, user, status, domain or None, text or None)
        return frame(rows, COLUMNS), _badge(rows)

    refresh.click(guarded(list_skills, 2), [status, domain, text], [table, badge], api_name="list_skills")
    status.change(guarded(list_skills, 2), [status, domain, text], [table, badge], api_name=False)

    skill_id = gr.Textbox(label="Skill ID", elem_id="skill_id")

    def pick_skill(evt: gr.SelectData):
        return evt.row_value[0]

    table.select(pick_skill, None, skill_id, api_name=False)

    with gr.Row():
        detail_button = gr.Button("Show detail")
        flag_reason = gr.Textbox(label="Flag reason")
        flag_button = gr.Button("Flag skill")
    detail = gr.JSON(label="Detail")

    def skill_detail(skill_id, request: gr.Request):
        user = current_user(ctx, request)
        return jsonable(handlers.skill_detail(ctx, user, skill_id))

    def flag_skill(skill_id, reason, request: gr.Request):
        user = current_user(ctx, request)
        handlers.flag_skill(ctx, user, skill_id, reason)
        gr.Info("Skill flagged")
        return jsonable(handlers.skill_detail(ctx, user, skill_id))

    detail_button.click(guarded(skill_detail), skill_id, detail, api_name="skill_detail")
    flag_button.click(guarded(flag_skill), [skill_id, flag_reason], detail, api_name="flag_skill")

    with gr.Accordion("Propose a skill", open=False):
        applicability = gr.Textbox(label="Applicability", lines=2)
        design_template = gr.Textbox(label="Design template", lines=4)
        preconditions = gr.Textbox(label="Preconditions")
        skill_domain = gr.Textbox(label="Skill domain")
        propose_button = gr.Button("Propose skill")
        proposed = gr.Textbox(label="Proposed skill ID", interactive=False)

    def propose_skill(applicability, design_template, preconditions, skill_domain, request: gr.Request):
        user = current_user(ctx, request)
        new_id = handlers.propose_skill(ctx, user, applicability, design_template, preconditions, skill_domain)
        gr.Info("Skill proposed")
        return new_id

    propose_button.click(
        guarded(propose_skill),
        [applicability, design_template, preconditions, skill_domain],
        proposed,
        api_name="propose_skill",
    )

    with gr.Accordion("Retrieval preview", open=False):
        hypothesis = gr.Textbox(label="Hypothesis")
        preview_button = gr.Button("Preview retrieval")
        preview = gr.JSON(label="Retrieved skills", elem_id="retrieval_preview")

    def retrieval_preview(hypothesis, request: gr.Request):
        user = current_user(ctx, request)
        return jsonable(handlers.retrieval_preview(ctx, user, hypothesis))

    preview_button.click(guarded(retrieval_preview), hypothesis, preview, api_name="retrieval_preview")

    with gr.Group(visible=False) as admin_box:
        gr.Markdown("Admin")
        verdict = gr.Radio(["approve", "reject"], value="approve", label="Library decision")
        reason = gr.Textbox(label="Decision reason")
        with gr.Row():
            decide_button = gr.Button("Record decision")
            restore_button = gr.Button("Restore")
            delete_button = gr.Button("Delete", variant="stop")
        with gr.Row():
            edit_applicability = gr.Textbox(label="New applicability")
            edit_design_template = gr.Textbox(label="New design template")
            edit_preconditions = gr.Textbox(label="New preconditions")
            edit_domain = gr.Textbox(label="New domain")
        edit_button = gr.Button("Save edits")

    def set_library_approval(skill_id, verdict, reason, request: gr.Request):
        user = current_user(ctx, request)
        handlers.set_library_approval(ctx, user, skill_id, verdict == "approve", reason)
        gr.Info("Decision recorded")
        return jsonable(handlers.skill_detail(ctx, user, skill_id))

    def restore_skill(skill_id, request: gr.Request):
        user = current_user(ctx, request)
        handlers.restore_skill(ctx, user, skill_id)
        gr.Info("Skill restored")
        return jsonable(handlers.skill_detail(ctx, user, skill_id))

    def delete_skill(skill_id, request: gr.Request):
        user = current_user(ctx, request)
        handlers.delete_skill(ctx, user, skill_id)
        gr.Info("Skill deleted")
        return {}

    def edit_skill(skill_id, applicability, design_template, preconditions, domain, request: gr.Request):
        user = current_user(ctx, request)
        fields = {
            name: value
            for name, value in (
                ("applicability", applicability),
                ("design_template", design_template),
                ("preconditions", preconditions),
                ("domain", domain),
            )
            if value
        }
        handlers.edit_skill(ctx, user, skill_id, **fields)
        gr.Info("Skill updated")
        return jsonable(handlers.skill_detail(ctx, user, skill_id))

    decide_button.click(
        guarded(set_library_approval), [skill_id, verdict, reason], detail, api_name="set_library_approval"
    )
    restore_button.click(guarded(restore_skill), skill_id, detail, api_name="restore_skill")
    delete_button.click(guarded(delete_skill), skill_id, detail, api_name="delete_skill")
    edit_button.click(
        guarded(edit_skill),
        [skill_id, edit_applicability, edit_design_template, edit_preconditions, edit_domain],
        detail,
        api_name="edit_skill",
    )
    return {"admin_boxes": [admin_box], "load": (list_skills, [status, domain, text], [table, badge])}
