import pandas as pd

from popper_upgrade.skills.context import describe_datasets


def test_describe_datasets_strips_prefix():
    frame = pd.DataFrame({"gender": ["f", "m"], "score": [1.5, 2.5]})

    [schema] = describe_datasets({"df_winobias": frame})

    assert schema.name == "winobias"
    assert schema.columns == [("gender", "object"), ("score", "float64")]
    assert schema.n_rows == 2


def test_describe_datasets_skips_unloadable_tables():
    assert describe_datasets({"df_broken": None}) == []
