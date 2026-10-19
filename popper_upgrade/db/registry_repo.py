import hashlib
import json
import uuid
from pathlib import Path

MODEL_COLUMNS = (
    "id::text as id, name, backend, served_model_id, size_class, environment, port, default_temperature, "
    "base_url, api_key_env, strip_tools, min_disk_gb"
)


def prompt_hash(role, text):
    return hashlib.sha256(f"{role}\n{text}".encode()).hexdigest()


class RegistryRepository:
    def __init__(self, conn):
        self._conn = conn

    def seed_from(self, directory):
        directory = Path(directory)
        for model in json.loads((directory / "models.json").read_text(encoding="utf-8")):
            self._conn.execute(
                "insert into registry.models (id, name, backend, served_model_id, size_class, environment, port, "
                "default_temperature, base_url, api_key_env, strip_tools, min_disk_gb) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (name) do update set backend = excluded.backend, "
                "served_model_id = excluded.served_model_id, size_class = excluded.size_class, "
                "environment = excluded.environment, port = excluded.port, "
                "default_temperature = excluded.default_temperature, base_url = excluded.base_url, "
                "api_key_env = excluded.api_key_env, strip_tools = excluded.strip_tools, "
                "min_disk_gb = excluded.min_disk_gb",
                (
                    str(uuid.uuid4()),
                    model["name"],
                    model["backend"],
                    model["served_model_id"],
                    model["size_class"],
                    model["environment"],
                    model.get("port"),
                    model.get("default_temperature"),
                    model.get("base_url"),
                    model.get("api_key_env"),
                    model.get("strip_tools", True),
                    model.get("min_disk_gb"),
                ),
            )
        sets = json.loads((directory / "hypotheses.json").read_text(encoding="utf-8"))
        for name, hypotheses in sets.items():
            set_id = self._conn.execute(
                "insert into registry.hypothesis_sets (id, name) values (%s, %s) "
                "on conflict (name) do update set name = excluded.name returning id",
                (str(uuid.uuid4()), name),
            ).fetchone()["id"]
            for hypothesis in hypotheses:
                self._conn.execute(
                    "insert into registry.hypotheses (id, set_id, text, split, ground_truth) "
                    "values (%s, %s, %s, %s, %s) on conflict (set_id, text) do update set "
                    "split = excluded.split, ground_truth = excluded.ground_truth",
                    (
                        str(uuid.uuid4()),
                        set_id,
                        hypothesis["text"],
                        hypothesis.get("split", "none"),
                        hypothesis.get("ground_truth"),
                    ),
                )
        for dataset in json.loads((directory / "datasets.json").read_text(encoding="utf-8")):
            self._conn.execute(
                "insert into registry.datasets (id, name, path, loader_type, schema, permutable_columns) "
                "values (%s, %s, %s, %s, %s::jsonb, %s::jsonb) on conflict (name) do update set "
                "path = excluded.path, loader_type = excluded.loader_type, schema = excluded.schema, "
                "permutable_columns = excluded.permutable_columns",
                (
                    str(uuid.uuid4()),
                    dataset["name"],
                    dataset["path"],
                    dataset["loader_type"],
                    json.dumps(dataset["schema"]),
                    json.dumps(dataset["permutable_columns"]),
                ),
            )

    def models(self, environment=None):
        if environment is None:
            return self._conn.execute(f"select {MODEL_COLUMNS} from registry.models order by name").fetchall()
        return self._conn.execute(
            f"select {MODEL_COLUMNS} from registry.models where environment = %s order by name", (environment,)
        ).fetchall()

    def model(self, name):
        row = self._conn.execute(f"select {MODEL_COLUMNS} from registry.models where name = %s", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        return row

    def hypothesis_set(self, name):
        rows = self._conn.execute(
            "select h.id::text as id, h.text, h.split, h.ground_truth from registry.hypotheses h "
            "join registry.hypothesis_sets s on s.id = h.set_id where s.name = %s order by h.text",
            (name,),
        ).fetchall()
        if not rows:
            raise KeyError(name)
        return rows

    def dataset(self, name):
        row = self._conn.execute(
            "select id::text as id, name, path, loader_type, schema, permutable_columns "
            "from registry.datasets where name = %s",
            (name,),
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return row

    def prompt_version(self, role, text):
        digest = prompt_hash(role, text)
        row = self._conn.execute(
            "insert into registry.prompt_versions (id, role, text, hash) values (%s, %s, %s, %s) "
            "on conflict (hash) do update set hash = excluded.hash returning id::text as id",
            (str(uuid.uuid4()), role, text, digest),
        ).fetchone()
        return row["id"]
