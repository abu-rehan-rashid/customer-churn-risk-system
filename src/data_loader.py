import os
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = "data/churn_data.csv"


def download_data():
    os.makedirs("data", exist_ok=True)
    print("Downloading Telco Churn dataset...")
    
    try:
        response = requests.get(DATA_URL, timeout=15)
        response.raise_for_status()
        
        # Parse ' ' whitespace strings as NaN during initial download
        df = pd.read_csv(DATA_URL, na_values=" ")
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset successfully saved to {DATA_PATH}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def inspect_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file missing at {DATA_PATH}. Run download_data() first.")
        
    df = pd.read_csv(DATA_PATH)
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nTarget Distribution (%):")
    print(df['Churn'].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    download_data()
    inspect_data()