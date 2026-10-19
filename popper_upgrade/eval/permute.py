from popper.utils import CustomDataLoader


def custom_factory(data_dir, permutable, seed):
    def build():
        loader = CustomDataLoader(data_folder=data_dir, random_seed=0 if seed is None else seed)
        if seed is not None:
            for table, columns in permutable.items():
                loader.permute_columns(table, list(columns))
            loader.data_desc = loader._generate_data_description()
        return loader

    return build
