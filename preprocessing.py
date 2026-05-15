"""
Data preprocessing module for Pakistan House Price Estimation
Handles data cleaning, missing value imputation, feature engineering, and encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re


def parse_area(area_str):
    """
    Parse area string and convert to square feet.
    Handles both Marla and Kanal units.
    1 Marla = 272.251 sq ft
    1 Kanal = 5445.02 sq ft
    """
    if pd.isna(area_str):
        return np.nan
    
    area_str = str(area_str).strip()
    
    # Extract numeric value using regex
    match = re.search(r'(\d+\.?\d*)', area_str)
    if not match:
        return np.nan
    
    numeric_value = float(match.group(1))
    
    # Determine unit and convert to square feet
    if 'Marla' in area_str or 'marla' in area_str:
        return numeric_value * 272.251
    elif 'Kanal' in area_str or 'kanal' in area_str:
        return numeric_value * 5445.02
    else:
        return np.nan


def preprocess_data(df):
    """
    Preprocess the dataset for model training.
    
    Steps:
    1. Drop duplicates and irrelevant columns
    2. Handle missing values (numeric: median, categorical: mode)
    3. Convert area to square feet
    4. Encode categorical columns
    5. Remove outliers using IQR method
    6. Log-transform price column
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (cleaned_df, feature_list, label_encoders_dict)
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # 1. Drop duplicates
    df = df.drop_duplicates(subset=['property_id'])
    
    # 2. Drop irrelevant columns
    cols_to_drop = ['property_id', 'location_id', 'page_url', 'date_added', 'agency', 'agent']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 3. Handle missing values
    # Numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # 4. Convert area to square feet
    df['area'] = df['area'].apply(parse_area)
    
    # Handle any remaining NaN in area (fill with median)
    df['area'] = df['area'].fillna(df['area'].median())
    
    # 5. Remove outliers using IQR method on price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    # 6. Log-transform price (target variable)
    df['price'] = np.log1p(df['price'])
    
    # 7. Encode categorical columns
    label_encoders = {}
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = {
            'encoder': le,
            'classes': le.classes_.tolist()
        }
    
    # 8. Get feature list (all columns except target)
    feature_list = [col for col in df.columns if col != 'price']
    
    return df, feature_list, label_encoders


if __name__ == "__main__":
    # Test the preprocessing function
    test_df = pd.read_csv('data/Pakistan House Prices and Property Listings.csv')
    print("Original shape:", test_df.shape)
    
    cleaned_df, features, encoders = preprocess_data(test_df)
    print("Cleaned shape:", cleaned_df.shape)
    print("\nFeatures:", features)
    print("\nEncoders dict keys:", encoders.keys())
    print("\nCleaned data head:")
    print(cleaned_df.head())
