alter table registry.models drop constraint if exists models_backend_check;
alter table registry.models
    add constraint models_backend_check
    check (backend in ('ollama', 'vllm', 'anthropic', 'openai', 'openai_compatible'));

alter table registry.models add column base_url text;
alter table registry.models add column api_key_env text;
alter table registry.models add column strip_tools boolean not null default true;

alter table registry.models
    add constraint models_base_url_check
    check (backend <> 'openai_compatible' or (base_url is not null and base_url <> ''));
