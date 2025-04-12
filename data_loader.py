import os

def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" File not found: {filepath}. Please upload it to the app's root directory.")
    return pd.read_csv(filepath)
