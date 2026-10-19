import os
from pathlib import Path

import gradio as gr

from popper_upgrade.app.auth import authenticate
from popper_upgrade.app.context import REGISTRY_DIR, AppContext
from popper_upgrade.app.handlers.launch import LaunchManager
from popper_upgrade.app.tabs import admin as admin_tab
from popper_upgrade.app.tabs import curation as curation_tab
from popper_upgrade.app.tabs import launch as launch_tab
from popper_upgrade.app.tabs import run_detail as run_detail_tab
from popper_upgrade.app.tabs import runs as runs_tab
from popper_upgrade.app.tabs import skills as skills_tab
from popper_upgrade.app.tabs.common import current_user, guarded
from popper_upgrade.db.connection import DEFAULT_DSN, connect
from popper_upgrade.db.migrate import migrate
from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.models import COMPATIBLE_BACKEND, popper_kwargs_for, serving_for
from popper_upgrade.run import default_trajectory_dir
from popper_upgrade.skills.curator import SkillCurator
from popper_upgrade.skills.embedding import Embedder

PORT = 7860


def build_app(ctx, launcher=None):
    launcher = launcher or LaunchManager(ctx)
    run_id = gr.Textbox(label="Run ID", render=False, elem_id="run_id")
    with gr.Blocks(title="POPPER control panel") as demo:
        gr.Markdown("# POPPER control panel")
        with gr.Tabs():
            with gr.Tab("Skills"):
                skills = skills_tab.build(ctx)
            with gr.Tab("Curation"):
                curation = curation_tab.build(ctx)
            with gr.Tab("Runs"):
                runs = runs_tab.build(ctx, run_id)
            with gr.Tab("Run detail"):
                run_detail_tab.build(ctx, run_id)
            with gr.Tab("Launch"):
                launch = launch_tab.build(ctx, launcher, curation["token"])
            with gr.Tab("Admin", visible=False) as admin_tab_item:
                admin = admin_tab.build(ctx)

        admin_boxes = skills["admin_boxes"]

        def show_admin(request: gr.Request):
            visible = current_user(ctx, request).role == "admin"
            return [gr.update(visible=visible) for _ in range(1 + len(admin_boxes))]

        demo.load(guarded(show_admin, 1 + len(admin_boxes)), None, [admin_tab_item, *admin_boxes], api_name=False)
        for fn, inputs, outputs in (skills["load"], runs["load"]):
            demo.load(guarded(fn, len(outputs)), inputs, outputs, api_name=False)
        for fn, outputs in (curation["refresh"], launch["refresh"]):
            demo.load(guarded(fn, len(outputs)), None, outputs, api_name=False)
        for fn, inputs, outputs in admin["load"]:
            demo.load(guarded(fn), inputs, outputs, api_name=False)
    return demo


def make_curator_factory(dsn):
    served = {}

    def serving_port(model):
        if model["backend"] != COMPATIBLE_BACKEND:
            return None
        if model["name"] not in served:
            context = serving_for(model)
            served[model["name"]] = (context, context.__enter__().port)
        return served[model["name"]][1]

    def factory(model_name):
        with connect(dsn) as conn:
            model = RegistryRepository(conn).model(model_name)
        kwargs = popper_kwargs_for(model, serving_port(model))
        return SkillCurator(kwargs["llm"], port=kwargs.get("server_port"), api_key=kwargs.get("api_key", "EMPTY"))

    return factory


def credentials_check(ctx):
    def check(username, password):
        with ctx.conn() as conn:
            return authenticate(conn, username, password) is not None

    return check


def main():
    dsn = os.environ.get("POPPER_DB_DSN", DEFAULT_DSN)
    with connect(dsn) as conn:
        migrate(conn)
        RegistryRepository(conn).seed_from(REGISTRY_DIR)
    ctx = AppContext(
        dsn,
        Embedder(),
        Path(default_trajectory_dir()),
        curator_factory=make_curator_factory(dsn),
        default_curator_model=os.environ.get("POPPER_CURATOR_MODEL"),
    )
    build_app(ctx).queue().launch(
        server_name=os.environ.get("POPPER_APP_HOST", "127.0.0.1"),
        server_port=PORT,
        share=False,
        auth=credentials_check(ctx),
    )


if __name__ == "__main__":
    main()
