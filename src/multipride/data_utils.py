import pandas as pd
import datasets as ds


def load_split(
    file_format: str,
    file_path: str,
    split: bool,
    test: bool,
    columns: list[str],
    val_size: float = 0.15,
    seed: int = 42,
    label_column: str = "label",
) -> ds.Dataset | ds.DatasetDict:
    data = ds.load_dataset(file_path)

    data = data.select_columns(columns)

    if not test:
        data = data.cast_column(
            label_column, ds.ClassLabel(names=["negative", "positive"])
        )

    if split:
        # split the dataset into train and val sets
        data_dict = data.train_test_split(
            test_size=val_size, shuffle=True, seed=seed, stratify_by_column=label_column
        )

        return data_dict

    else:
        return data


def augment_data(
    train_set: ds.Dataset, label_column: str = "label", augmentation_type: str = "ros"
):
    df = train_set.to_pandas()

    if augmentation_type == "ros":
        print(f"Original dataset shape: {df.shape}")
        print(f"Original class distribution:\n{df[label_column].value_counts()}")

        # 1. Separate the classes
        df_majority = df[df[label_column] == 0]
        df_minority = df[df[label_column] == 1]

        # 2. Determine the target size (size of the majority class)
        target_size = len(df_majority) - len(df_minority)

        print(
            f"Oversampling minority class from {len(df_minority)} to {len(df_majority)}..."
        )

        # 3. Oversample the minority class *with replacement*
        df_minority_oversampled = df_minority.sample(
            n=target_size,
            replace=True,  # This is the key part of over-sampling
            random_state=42,  # For reproducibility
        )

        # 4. Combine the original dataset with the sample from the minority class
        df_oversampled = pd.concat([df, df_minority_oversampled])

        # 5. Shuffle the combined dataframe
        # This is CRITICAL. The Trainer needs shuffled data.
        df_oversampled_shuffled = ds.shuffle(df_oversampled, random_state=42)

        print(f"New oversampled dataset shape: {df_oversampled_shuffled.shape}")
        print(
            f"New class distribution:\n{df_oversampled_shuffled[label_column].value_counts()}"
        )

        data = df_oversampled_shuffled.reset_index(drop=True)

        return ds.from_pandas(data)
