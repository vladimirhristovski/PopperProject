alter table runs.runs add column cost_usd double precision check (cost_usd is null or cost_usd >= 0);

alter table registry.models add column pricing jsonb;
alter table registry.models add column budget_eur double precision check (budget_eur is null or budget_eur > 0);

alter table registry.models drop constraint if exists models_backend_check;
alter table registry.models
    add constraint models_backend_check
    check (backend in ('ollama', 'vllm', 'anthropic', 'openai', 'openai_compatible', 'anthropic_messages'));

alter table registry.models drop constraint if exists models_base_url_check;
alter table registry.models
    add constraint models_base_url_check
    check (backend not in ('openai_compatible', 'anthropic_messages') or (base_url is not null and base_url <> ''));
