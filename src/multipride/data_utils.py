import pandas as pd
import datasets as ds


def load_aug_split(
    file_format: str,
    file_path: str,
    split: bool,
    test: bool,
    augmentation: str,
    columns: list[str],
    val_size: float,
    seed: int,
) -> ds.Dataset | ds.DatasetDict:
    df = pd.read_csv(file_path)

    if augmentation == "ros":
        pass

    data = ds.from_pandas(df)

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
