import os
import glob
import pandas as pd
from github import Github, GithubException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
REPO_OWNER = GITHUB_USERNAME
DATA_DIR = "tourism_project/data"

# Ensure you have your GitHub Personal Access Token saved in environment variables
GITHUB_TOKEN = os.getenv("GH_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

# Construct the raw URL to pull data from your GitHub repository
DATASET_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/tourism_project/data/tourism.csv"

def prepare_data():
    print("Step 1: Loading dataset from GitHub...")
    try:
        df = pd.read_csv(DATASET_URL)
        print("Dataset loaded from GitHub successfully.")
    except Exception as e:
        print(f"Failed to load dataset from URL {DATASET_URL}. Error: {e}")
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
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f" Encoded column: {col}")

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
    os.makedirs(DATA_DIR, exist_ok=True)
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
