import datasets as ds


def load_split(file_format, file_path, split, test, columns, val_size, seed):
    data = ds.load_dataset(file_format, data_files=file_path, split="train")

    data = data.select_columns(columns)

    if not test:
        data = data.cast_column("label", ds.ClassLabel(names=["negative", "positive"]))

    if split:
        # split the dataset into train and val sets
        data_dict = data.train_test_split(
            test_size=val_size, shuffle=True, seed=42, stratify_by_column="label"
        )

        return data_dict

    else:
        return data
