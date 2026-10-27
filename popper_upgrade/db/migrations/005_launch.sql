alter table registry.models add column quantization text;
alter table registry.models add column dtype text;
alter table registry.models add column max_model_len integer check (max_model_len is null or max_model_len > 0);
alter table registry.models add column system_prefix text;
alter table registry.models add column request_overrides jsonb;
alter table registry.models add column serving jsonb;
