from dataclasses import dataclass


@dataclass
class DatasetSchema:
    name: str
    columns: list[tuple[str, str]]
    n_rows: int


@dataclass
class RunContext:
    hypotheses: list[str]
    domain: str
    datasets: list[DatasetSchema]


def describe_datasets(table_dict):
    return [
        DatasetSchema(
            name=name.removeprefix("df_"),
            columns=[(str(column), str(dtype)) for column, dtype in frame.dtypes.items()],
            n_rows=len(frame),
        )
        for name, frame in table_dict.items()
        if frame is not None
    ]
