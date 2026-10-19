alter table runs.runs add column answer boolean;
alter table runs.runs add column task text;
alter table runs.runs add column replicate integer;
alter table runs.runs add column control_degenerate boolean not null default false;

create index runs_experiment_idx on runs.runs (experiment_id);

create table registry.curation_references (
    id uuid primary key,
    skill_id uuid not null references skills.skills (id),
    context_hash text not null,
    verdict text not null check (verdict in ('approve', 'reject')),
    reason text not null default '',
    labelled_by uuid references auth.users (id),
    created_at timestamptz not null default now()
);
