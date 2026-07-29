import os
import pandas as pd


DATA_PATH = r"D:\sem4\mlProject\placement_Dataset.csv"



def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist.")

    df = pd.read_csv(path)
    return df



def get_data_summary(path:str = DATA_PATH) -> dict:
    df = load_data()
    summary = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isnull().sum().to_dict(),
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


if __name__ == "__main__":
    df = load_data()
    print(get_data_summary())

