#%%writefile tourism_project/model_building/prep.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/deployment"
DATA_PATH = os.path.join(DATA_DIR, "tourism.csv")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

def prepare_data():
    print("Step 1: Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset from URL {DATA_PATH}. Error: {e}")
        return

    print("Step 2: Performing data cleaning...")
    # Remove unnecessary columns
    cols_to_drop = ["CustomerID", "Unnamed: 0"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Standardize Categorical Values
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    # Label Encoding for categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f" Encoded column: {col}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Encoders saved to {ENCODERS_PATH}")

    print("Step 3: Splitting into train and test sets...")
    target_col = "ProdTaken"

    if target_col not in df.columns:
        print(f"Target column '{target_col}' missing from data!")
        return

    # Split into X (features) and y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Perform stratified train-test split
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save locally
    paths = {
        "Xtrain.csv": os.path.join(DATA_DIR, "Xtrain.csv"),
        "Xtest.csv": os.path.join(DATA_DIR, "Xtest.csv"),
        "ytrain.csv": os.path.join(DATA_DIR, "ytrain.csv"),
        "ytest.csv": os.path.join(DATA_DIR, "ytest.csv"),
    }

    # Save mapping logic safely without dangerous eval statements
    data_mapping = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}

    for filename, path in paths.items():
        obj = data_mapping[filename]
        if isinstance(obj, pd.Series):
            obj.to_csv(path, index=False, header=True)
        else:
            obj.to_csv(path, index=False)

    print(f"Local files saved in {DATA_DIR}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    prepare_data()
