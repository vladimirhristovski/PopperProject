alter table registry.models add column revision text check (revision is null or revision ~ '^[0-9a-f]{40}$');
