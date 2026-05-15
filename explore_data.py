import pandas as pd
import numpy as np

# Load the dataset
csv_path = r'data/Pakistan House Prices and Property Listings.csv'
df = pd.read_csv(csv_path)

print("="*80)
print("DATASET EXPLORATION")
print("="*80)

print("\n1. SHAPE AND SIZE:")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. COLUMN NAMES AND DATA TYPES:")
print(df.dtypes)

print("\n3. FIRST 5 ROWS:")
print(df.head())

print("\n4. MISSING VALUES:")
print(df.isnull().sum())

print("\n5. DATASET INFO:")
df.info()

print("\n6. STATISTICAL SUMMARY:")
print(df.describe())

print("\n7. UNIQUE VALUES IN KEY COLUMNS:")
for col in df.columns:
    if df[col].dtype == 'object':
        unique_count = df[col].nunique()
        if unique_count <= 50:
            print(f"   {col}: {unique_count} unique values")
        else:
            print(f"   {col}: {unique_count} unique values (showing first 10)")
            print(f"      {df[col].unique()[:10].tolist()}")
