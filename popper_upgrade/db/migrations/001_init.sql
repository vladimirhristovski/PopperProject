create extension if not exists vector;

create schema if not exists auth;
create schema if not exists skills;
create schema if not exists runs;
create schema if not exists registry;
create schema if not exists audit;

create table auth.users (
    id uuid primary key,
    username text not null unique,
    password_hash text not null,
    role text not null check (role in ('user', 'admin')),
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table auth.sessions (
    id uuid primary key,
    user_id uuid not null references auth.users (id),
    created_at timestamptz not null default now(),
    expires_at timestamptz not null
);

create table registry.models (
    id uuid primary key,
    name text not null unique,
    backend text not null check (backend in ('ollama', 'vllm', 'anthropic', 'openai')),
    served_model_id text not null,
    size_class text not null check (size_class in ('small', 'large')),
    environment text not null check (environment in ('laptop', 'cluster')),
    port integer,
    default_temperature real
);

create table registry.hypothesis_sets (
    id uuid primary key,
    name text not null unique,
    created_at timestamptz not null default now()
);

create table registry.hypotheses (
    id uuid primary key,
    set_id uuid not null references registry.hypothesis_sets (id),
    text text not null,
    split text not null check (split in ('train', 'test', 'none')),
    ground_truth boolean,
    unique (set_id, text)
);

create table registry.datasets (
    id uuid primary key,
    name text not null unique,
    path text not null,
    loader_type text not null,
    schema jsonb not null,
    permutable_columns jsonb not null
);

create table registry.prompt_versions (
    id uuid primary key,
    role text not null check (role in ('curator', 'reviewer', 'extractor')),
    text text not null,
    hash text not null unique,
    created_at timestamptz not null default now()
);

create table registry.library_snapshots (
    id uuid primary key,
    created_at timestamptz not null default now(),
    embedding_model text not null,
    skill_ids jsonb not null,
    content_hash text not null,
    path text
);

create table skills.skills (
    id uuid primary key,
    applicability text not null,
    design_template text not null,
    preconditions text not null default '',
    domain text not null default '',
    source_hypothesis text not null default '',
    source_run_id uuid,
    parent_skill_id uuid references skills.skills (id),
    approved boolean,
    decided_by text,
    decision_reason text,
    proposed_by uuid references auth.users (id),
    flagged boolean not null default false,
    times_used integer not null default 0,
    times_passed integer not null default 0,
    content_hash text not null,
    embedding_model text not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index skills_embedding_idx on skills.skills using hnsw (embedding vector_cosine_ops);

create table skills.skill_decisions (
    id uuid primary key,
    skill_id uuid not null references skills.skills (id),
    run_id uuid,
    trigger text not null check (trigger in ('entry', 'pre_run', 'admin')),
    decider text not null check (decider in ('curator', 'human')),
    user_id uuid references auth.users (id),
    verdict text not null check (verdict in ('approve', 'reject')),
    reason text not null,
    model text,
    prompt_version_id uuid references registry.prompt_versions (id),
    created_at timestamptz not null default clock_timestamp()
);

create table skills.skill_flags (
    id uuid primary key,
    skill_id uuid not null references skills.skills (id),
    user_id uuid not null references auth.users (id),
    reason text not null,
    created_at timestamptz not null default now(),
    resolved boolean not null default false
);

create table runs.experiments (
    id uuid primary key,
    name text not null,
    spec jsonb not null,
    status text not null,
    created_at timestamptz not null default now()
);

create table runs.runs (
    id uuid primary key,
    experiment_id uuid references runs.experiments (id),
    created_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null check (status in ('running', 'completed', 'error')),
    environment text not null check (environment in ('laptop', 'cluster')),
    launched_by uuid references auth.users (id),
    hypothesis_id uuid references registry.hypotheses (id),
    hypothesis text not null,
    arm text not null,
    skills_enabled boolean not null,
    reviewer_enabled boolean not null,
    human_review boolean not null,
    model_id uuid references registry.models (id),
    model_name text not null,
    temperature real,
    seed integer,
    alpha real not null,
    aggregate_test text not null,
    max_num_of_tests integer not null,
    max_retry integer not null,
    time_limit integer not null,
    domain text not null,
    datasets jsonb not null,
    permuted boolean not null default false,
    permute_seed integer,
    library_snapshot_id uuid references registry.library_snapshots (id),
    popper_version text not null,
    git_sha text not null,
    pins_hash text not null,
    res boolean,
    res_stat double precision,
    n_tests integer,
    parsed_result jsonb,
    reviewer_rejections integer,
    reviewer_overrides integer,
    tokens_in integer,
    tokens_out integer,
    wall_time_s real,
    trajectory_path text,
    error text
);

alter table skills.skills
    add constraint skills_source_run_fk foreign key (source_run_id) references runs.runs (id)
    deferrable initially deferred;
alter table skills.skill_decisions
    add constraint skill_decisions_run_fk foreign key (run_id) references runs.runs (id);

create table runs.run_skills (
    run_id uuid not null references runs.runs (id),
    skill_id uuid not null references skills.skills (id),
    role text not null check (
        role in ('curator_approved', 'curator_rejected', 'human_override_approve', 'human_override_reject', 'injected')
    ),
    round integer,
    similarity real,
    round_key integer generated always as (coalesce(round, 0)) stored,
    primary key (run_id, skill_id, role, round_key)
);

create table runs.design_reviews (
    id uuid primary key,
    run_id uuid not null references runs.runs (id),
    round integer not null,
    attempt integer not null,
    proposal text not null,
    approved boolean not null,
    reason text not null,
    overridden boolean not null default false,
    created_at timestamptz not null default now()
);

create table audit.events (
    id bigserial primary key,
    user_id uuid references auth.users (id),
    action text not null,
    target_type text not null,
    target_id text not null,
    before jsonb,
    after jsonb,
    created_at timestamptz not null default clock_timestamp()
);

create function audit.reject_mutation() returns trigger as $$
begin
    raise exception 'audit.events is append-only';
end;
$$ language plpgsql;

create trigger events_append_only
    before update or delete on audit.events
    for each row execute function audit.reject_mutation();
