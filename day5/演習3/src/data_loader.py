# day5/演習3/src/data_loader.py
import pandas as pd


def load_test_data(csv_path="data/Titanic.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y
