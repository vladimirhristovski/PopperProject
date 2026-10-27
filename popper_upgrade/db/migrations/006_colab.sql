alter table registry.models drop constraint if exists models_environment_check;
alter table registry.models
    add constraint models_environment_check
    check (environment in ('laptop', 'cluster', 'colab'));

alter table runs.runs drop constraint if exists runs_environment_check;
alter table runs.runs
    add constraint runs_environment_check
    check (environment in ('laptop', 'cluster', 'colab'));
