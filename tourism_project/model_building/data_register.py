"""
Data Registration Module
========================
Validates and registers the tourism dataset for the ML pipeline.
"""

from pathlib import Path
import pandas as pd

# --- Configuration & Paths ---
BASE_DIR = Path("tourism_project")
DATASET_FILE = BASE_DIR / "data" / "tourism.csv"

# --- File Validation ---
if not DATASET_FILE.is_file():
    raise FileNotFoundError(f"Critical Error: Target data file missing at '{DATASET_FILE}'")

# --- Data Ingestion ---
tourism_df = pd.read_csv(DATASET_FILE)

# --- Schema Verification ---
REQUIRED_FIELDS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

# Identify any absent features
absent_fields = [field for field in REQUIRED_FIELDS if field not in tourism_df.columns]

if absent_fields:
    raise ValueError(f"Schema Validation Failed! Missing fields: {absent_fields}")

# --- Execution Logs & Summary ---
total_rows, total_cols = tourism_df.shape

print("=" * 50)
print(" SUCCESS: DATASET REGISTERED ".center(50, "#"))
print("=" * 50)
print(f"• Location:   {DATASET_FILE}")
print(f"• Records:    {total_rows:,}")
print(f"• Features:   {total_cols}")
print("-" * 50)
print("Registered Features:")
print(f"  {', '.join(tourism_df.columns)}")
print("-" * 50)
print("Target Variable Distribution ('ProdTaken'):")
print(tourism_df["ProdTaken"].value_counts().to_string())
print("=" * 50)
