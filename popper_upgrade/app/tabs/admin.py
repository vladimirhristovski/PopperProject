import json

import gradio as gr

from popper_upgrade.app.handlers import admin as handlers
from popper_upgrade.app.tabs.common import current_user, frame, guarded, jsonable

USER_COLUMNS = ("id", "username", "role", "active", "created_at")
MODEL_COLUMNS = (
    "name",
    "backend",
    "served_model_id",
    "size_class",
    "environment",
    "port",
    "default_temperature",
    "base_url",
    "api_key_env",
    "strip_tools",
    "min_disk_gb",
    "quantization",
    "dtype",
    "max_model_len",
    "system_prefix",
    "request_overrides",
    "serving",
    "pricing",
    "budget_eur",
)
PROMPT_COLUMNS = ("id", "role", "hash", "created_at")


def _user_id(ctx, user, username):
    for row in handlers.list_users(ctx, user):
        if row["username"] == username:
            return row["id"]
    raise KeyError(username)


def build(ctx):
    with gr.Accordion("Users", open=True):
        users = gr.Dataframe(headers=list(USER_COLUMNS), interactive=False, elem_id="users_table")
        refresh_users = gr.Button("Refresh users")
        with gr.Row():
            new_username = gr.Textbox(label="New username")
            new_password = gr.Textbox(label="New password", type="password")
            new_role = gr.Radio(["user", "admin"], value="user", label="New role")
            create_button = gr.Button("Create user")
        with gr.Row():
            target_username = gr.Textbox(label="Existing username")
            target_role = gr.Radio(["user", "admin"], value="user", label="Role")
            role_button = gr.Button("Set role")
            active = gr.Checkbox(label="Active", value=True)
            active_button = gr.Button("Set active")

    def list_users(request: gr.Request):
        user = current_user(ctx, request)
        return frame(handlers.list_users(ctx, user), USER_COLUMNS)

    def create_user_h(username, password, role, request: gr.Request):
        user = current_user(ctx, request)
        handlers.create_user_h(ctx, user, username, password, role)
        gr.Info("User created")
        return frame(handlers.list_users(ctx, user), USER_COLUMNS)

    def set_role_h(username, role, request: gr.Request):
        user = current_user(ctx, request)
        handlers.set_role_h(ctx, user, _user_id(ctx, user, username), role)
        return frame(handlers.list_users(ctx, user), USER_COLUMNS)

    def set_active_h(username, active, request: gr.Request):
        user = current_user(ctx, request)
        handlers.set_active_h(ctx, user, _user_id(ctx, user, username), active)
        return frame(handlers.list_users(ctx, user), USER_COLUMNS)

    refresh_users.click(guarded(list_users), None, users, api_name="list_users")
    create_button.click(guarded(create_user_h), [new_username, new_password, new_role], users, api_name="create_user_h")
    role_button.click(guarded(set_role_h), [target_username, target_role], users, api_name="set_role_h")
    active_button.click(guarded(set_active_h), [target_username, active], users, api_name="set_active_h")

    with gr.Accordion("Models", open=False):
        models = gr.Dataframe(headers=list(MODEL_COLUMNS), interactive=False)
        with gr.Row():
            model_name = gr.Textbox(label="Name")
            backend = gr.Dropdown(
                ["ollama", "vllm", "anthropic", "openai", "openai_compatible", "anthropic_messages"],
                value="ollama",
                label="Backend",
            )
            served_model_id = gr.Textbox(label="Served model id")
            size_class = gr.Radio(["small", "large"], value="small", label="Size class")
            environment = gr.Radio(["laptop", "cluster"], value="laptop", label="Environment")
            port = gr.Number(label="Port", precision=0)
            temperature = gr.Number(label="Default temperature")
        with gr.Row():
            base_url = gr.Textbox(label="Base URL (openai_compatible)")
            api_key_env = gr.Textbox(label="API key env var (openai_compatible, optional)")
            strip_tools = gr.Checkbox(label="Strip tools (openai_compatible)", value=True)
            min_disk_gb = gr.Number(label="Min free disk GB (cluster)", precision=0)
        with gr.Row():
            quantization = gr.Textbox(label="vLLM quantization (empty = awq, none = off)")
            dtype = gr.Textbox(label="vLLM dtype (empty = float16)")
            max_model_len = gr.Number(label="Max model length", precision=0)
            system_prefix = gr.Textbox(label="System prefix")
        with gr.Row():
            request_overrides = gr.Textbox(label="Request overrides (JSON: drop, set, rename)")
            serving = gr.Textbox(label="Serving facts (JSON: vllm_args, weights_gb, kv_kb_per_token, gated, fallback)")
        with gr.Row():
            pricing = gr.Textbox(label="Pricing (JSON, USD per 1M: input, cache_read, cache_write, output)")
            budget_eur = gr.Number(label="Budget EUR (paid models)")
        model_button = gr.Button("Save model")

    def upsert_model(
        name,
        backend,
        served_model_id,
        size_class,
        environment,
        port,
        temperature,
        base_url,
        api_key_env,
        strip_tools,
        min_disk_gb,
        quantization,
        dtype,
        max_model_len,
        system_prefix,
        request_overrides,
        serving,
        pricing,
        budget_eur,
        request: gr.Request,
    ):
        user = current_user(ctx, request)
        handlers.upsert_model(
            ctx,
            user,
            name,
            backend,
            served_model_id,
            size_class,
            environment,
            int(port) if port is not None else None,
            temperature,
            base_url,
            api_key_env,
            strip_tools,
            int(min_disk_gb) if min_disk_gb is not None else None,
            quantization,
            dtype,
            int(max_model_len) if max_model_len is not None else None,
            system_prefix,
            json.loads(request_overrides) if request_overrides else None,
            json.loads(serving) if serving else None,
            json.loads(pricing) if pricing else None,
            budget_eur,
        )
        gr.Info("Model saved")
        return frame(handlers.list_models(ctx, user), MODEL_COLUMNS)

    def load_models(request: gr.Request):
        user = current_user(ctx, request)
        if user.role != "admin":
            return gr.update()
        return frame(handlers.list_models(ctx, user), MODEL_COLUMNS)

    def load_users(request: gr.Request):
        user = current_user(ctx, request)
        if user.role != "admin":
            return gr.update()
        return frame(handlers.list_users(ctx, user), USER_COLUMNS)

    model_button.click(
        guarded(upsert_model),
        [
            model_name,
            backend,
            served_model_id,
            size_class,
            environment,
            port,
            temperature,
            base_url,
            api_key_env,
            strip_tools,
            min_disk_gb,
            quantization,
            dtype,
            max_model_len,
            system_prefix,
            request_overrides,
            serving,
            pricing,
            budget_eur,
        ],
        models,
        api_name="upsert_model",
    )

    with gr.Accordion("Prompt versions", open=False):
        prompts = gr.Dataframe(headers=list(PROMPT_COLUMNS), interactive=False)
        refresh_prompts = gr.Button("Refresh prompts")
        prompt_role = gr.Radio(["curator", "reviewer", "extractor"], value="curator", label="Role")
        prompt_text = gr.Textbox(label="Prompt text", lines=6)
        prompt_button = gr.Button("Add prompt version")

    def list_prompt_versions(request: gr.Request):
        user = current_user(ctx, request)
        return frame(handlers.list_prompt_versions(ctx, user), PROMPT_COLUMNS)

    def add_prompt_version(role, text, request: gr.Request):
        user = current_user(ctx, request)
        handlers.add_prompt_version(ctx, user, role, text)
        gr.Info("Prompt version added")
        return frame(handlers.list_prompt_versions(ctx, user), PROMPT_COLUMNS)

    refresh_prompts.click(guarded(list_prompt_versions), None, prompts, api_name="list_prompt_versions")
    prompt_button.click(guarded(add_prompt_version), [prompt_role, prompt_text], prompts, api_name="add_prompt_version")

    with gr.Accordion("Import run bundle", open=False):
        bundle_path = gr.Textbox(label="Bundle directory or archive path")
        bundle_button = gr.Button("Import bundle")
        archive_file = gr.File(label="Run archive (.tar.gz)", type="filepath", elem_id="archive_file")
        archive_button = gr.Button("Import archive")
        bundle_result = gr.JSON(label="Imported rows")

    def import_bundle_h(path, request: gr.Request):
        user = current_user(ctx, request)
        return jsonable(handlers.import_bundle_h(ctx, user, path))

    def import_archive_h(path, request: gr.Request):
        if not path:
            raise ValueError("choose an archive first")
        user = current_user(ctx, request)
        return jsonable(handlers.import_bundle_h(ctx, user, path))

    bundle_button.click(guarded(import_bundle_h), bundle_path, bundle_result, api_name="import_bundle_h")
    archive_button.click(guarded(import_archive_h), archive_file, bundle_result, api_name="import_archive_h")
    return {"load": [(load_users, None, users), (load_models, None, models)]}
