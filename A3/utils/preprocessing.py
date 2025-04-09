import os
import pathlib
import shutil
import random
import pandas as pd


def make_train_data(dir_name="Datasets", file_name="TrainData.csv"):
    csv_path = os.path.join(dir_name, file_name)
    data = pd.read_csv(csv_path)

    base_folder = os.path.join(dir_name, "train")
    os.makedirs(base_folder, exist_ok=True)

    for index, row in data.iterrows():
        category = str(row["Category"]).strip()
        text = row["Text"]

        category_folder = os.path.join(base_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        file_name = f"{index}.txt"
        file_path = os.path.join(category_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Files have been created in the respective category folders under {dir_name}/train.")


def make_test_data(dir_name="Datasets", file_name="TestLabels.csv"):
    csv_path = os.path.join(dir_name, file_name)
    data = pd.read_csv(csv_path)

    base_folder = os.path.join(dir_name, "test")
    os.makedirs(base_folder, exist_ok=True)

    for index, row in data.iterrows():
        category = str(row["Label - (business, tech, politics, sport, entertainment)"]).strip()
        text = row["Text"]

        category_folder = os.path.join(base_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        file_name = f"{index}.txt"
        file_path = os.path.join(category_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Files have been created in the respective category folders under {dir_name}/test.")


def train_val_split(dir_name="Datasets", split_ratio=0.2):
    base_dir = pathlib.Path(dir_name)
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"
    for category in ("sport", "tech", "business", "politics", "entertainment"):
        os.makedirs(val_dir / category, exist_ok=True)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)  # Shuffle the list of files
        num_val_samples = int(split_ratio * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)

    print("Created validation data.")

if __name__ == "__main__":
    make_train_data(dir_name="Datasets", file_name="TrainData.csv")
    make_test_data(dir_name="Datasets", file_name="TestLabels.csv")
    train_val_split(dir_name="Datasets", split_ratio=0.2)
