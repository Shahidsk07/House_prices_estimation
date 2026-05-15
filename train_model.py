"""
Model training module for Pakistan House Price Estimation
Trains XGBoost model with hyperparameter tuning and generates SHAP explanations
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from preprocessing import preprocess_data


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_model():
    """
    Train XGBoost model on preprocessed data.
    Saves model, encoders, features, and columns to models/ folder.
    Saves evaluation plots to plots/ folder.
    """
    
    print("="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv('data/Pakistan House Prices and Property Listings.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Preprocess
    df_cleaned, feature_list, encoders = preprocess_data(df)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Features: {feature_list}")
    
    # Save feature list and column names for later use
    Path('models').mkdir(exist_ok=True)
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(feature_list, f)
    
    with open('models/columns.pkl', 'wb') as f:
        pickle.dump(df_cleaned.columns.tolist(), f)
    
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print("\n" + "="*80)
    print("STEP 2: SPLITTING DATA")
    print("="*80)
    
    # Separate features and target
    X = df_cleaned[feature_list]
    y = df_cleaned['price']
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\n" + "="*80)
    print("STEP 3: TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Create and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Training XGBoost model with parameters:")
    print(f"  n_estimators: 500")
    print(f"  learning_rate: 0.05")
    print(f"  max_depth: 6")
    print(f"  subsample: 0.8")
    print(f"  colsample_bytree: 0.8")
    
    model.fit(X_train, y_train)
    print("✓ Model training completed!")
    
    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved to models/model.pkl")
    
    print("\n" + "="*80)
    print("STEP 4: MODEL EVALUATION")
    print("="*80)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert back from log scale to original price scale
    y_train_original = np.expm1(y_train.values)
    y_test_original = np.expm1(y_test.values)
    y_pred_train_original = np.expm1(y_pred_train)
    y_pred_test_original = np.expm1(y_pred_test)
    
    # Calculate metrics on test set
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    mae = mean_absolute_error(y_test_original, y_pred_test_original)
    r2 = r2_score(y_test_original, y_pred_test_original)
    mape = calculate_mape(y_test_original, y_pred_test_original)
    
    print("\nTEST SET METRICS:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:,.2f} PKR")
    print(f"  MAE (Mean Absolute Error): {mae:,.2f} PKR")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Training metrics for comparison
    rmse_train = np.sqrt(mean_squared_error(y_train_original, y_pred_train_original))
    r2_train = r2_score(y_train_original, y_pred_train_original)
    print(f"\nTRAIN SET METRICS:")
    print(f"  RMSE: {rmse_train:,.2f} PKR")
    print(f"  R² Score: {r2_train:.4f}")
    
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)
    
    Path('plots').mkdir(exist_ok=True)
    
    # 1. Feature Importance
    print("Generating feature importance plot...")
    feature_importance = pd.DataFrame({
        'feature': feature_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/feature_importance.png")
    
    # 2. Actual vs Predicted
    print("Generating actual vs predicted plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, y_pred_test_original, alpha=0.5, s=10)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (PKR)', fontsize=12)
    plt.ylabel('Predicted Price (PKR)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/actual_vs_predicted.png")
    
    # 3. Residual Plot
    print("Generating residual plot...")
    residuals = y_test_original - y_pred_test_original
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (PKR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/residual_plot.png")
    
    # 4. SHAP Analysis
    print("Generating SHAP plots (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_list, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/shap_summary.png")
    
    # SHAP Bar Plot (Mean Absolute)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_list, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/shap_bar.png")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✓ Model saved: models/model.pkl")
    print(f"✓ Encoders saved: models/encoders.pkl")
    print(f"✓ Features saved: models/features.pkl")
    print(f"✓ Columns saved: models/columns.pkl")
    print(f"✓ All plots saved to: plots/")
    
    return {
        'model': model,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }


if __name__ == "__main__":
    train_model()
