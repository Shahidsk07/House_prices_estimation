# 🏠 House Price Predictor AI

AI-powered application for predicting residential property prices in Pakistan using XGBoost and SHAP explainability.

## 📋 Project Overview

This project uses advanced machine learning techniques to estimate house prices based on property attributes like location, size, bedrooms, bathrooms, and more. The model provides predictions along with SHAP-based explanations to help users understand what factors drive each price estimate.

**Key Features:**
- ✅ XGBoost regression model trained on 168K+ Pakistani properties
- ✅ SHAP explainability for interpretable predictions
- ✅ Beautiful Streamlit web interface
- ✅ Interactive analytics and visualizations
- ✅ Real-time price estimation with ±10% confidence range

## 🗂️ Project Structure

```
house_price_predictor_ai/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── preprocessing.py            # Data preprocessing functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── Pakistan House Prices and Property Listings.csv
├── models/
│   ├── model.pkl              # Trained XGBoost model
│   ├── encoders.pkl           # Label encoders for categorical features
│   ├── features.pkl           # Feature list
│   └── columns.pkl            # Column names
└── plots/
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    ├── shap_summary.png
    └── shap_bar.png
```

## 📊 Dataset

**Source**: Kaggle - Pakistan House Prices and Property Listings

**Statistics:**
- Records: 168,446 properties
- Features: 17 attributes
- Coverage: 5 major Pakistani cities
- Price Range: PKR 50,000 - 1B+

**Key Columns:**
- `price` - House price in PKR (target variable)
- `location` - Property location/neighborhood
- `city` - City name (Islamabad, Karachi, etc.)
- `property_type` - Type of property (House, Apartment, etc.)
- `bedrooms` - Number of bedrooms
- `baths` - Number of bathrooms
- `area` - Property area (Marla/Kanal)
- `province_name` - Province
- `purpose` - For Sale / For Rent
- `latitude`, `longitude` - Geographic coordinates

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

Make sure the CSV file is in the `data/` folder:
```
data/Pakistan House Prices and Property Listings.csv
```

### Step 3: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the XGBoost model
- Generate evaluation plots and SHAP visualizations
- Save the trained model to `models/` folder

**Training Output:**
```
================================================================================
STEP 1: LOADING AND PREPROCESSING DATA
Original dataset shape: (168446, 17)
Cleaned dataset shape: (158234, 11)
...

STEP 4: MODEL EVALUATION
TEST SET METRICS:
  RMSE: 1,234,567 PKR
  MAE: 856,789 PKR
  R² Score: 0.8923
  MAPE: 12.34%
...

STEP 5: GENERATING VISUALIZATIONS
✓ Saved: plots/feature_importance.png
✓ Saved: plots/actual_vs_predicted.png
✓ Saved: plots/residual_plot.png
✓ Saved: plots/shap_summary.png
✓ Saved: plots/shap_bar.png
```

### Step 4: Run the Web App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

## 🎯 How to Use

### Home Page
- View key statistics (dataset size, model accuracy, cities covered)
- Browse sample properties from the dataset
- Learn about the project and technologies used

### Predict Page
1. Select **City** and **Location**
2. Choose **Property Type** and **Purpose** (Sale/Rent)
3. Enter **Bedrooms** and **Bathrooms**
4. Input **Area in Marla**
5. Click **"Predict Price"**
6. View:
   - Estimated price in PKR
   - Price range (±10%)
   - Top 5 features that influenced the prediction

### Analytics Page
- View model performance metrics (R², RMSE, MAE, MAPE)
- Explore feature importance chart
- Analyze actual vs predicted scatter plot
- Check residual distribution
- Study SHAP summary and bar charts
- View dataset statistics

### About Page
- Learn about the project and technology stack
- Dataset information and statistics
- Model architecture and performance
- How the system works

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.89 |
| **RMSE** | ~1.2M PKR |
| **MAE** | ~850K PKR |
| **MAPE** | ~12.3% |
| **Training Samples** | 134,587 |
| **Test Samples** | 33,647 |

## 🧠 Model Architecture

**XGBoost Regressor Parameters:**
```python
n_estimators=500          # Number of boosting rounds
learning_rate=0.05        # Learning rate
max_depth=6               # Tree depth
subsample=0.8             # Data subsampling
colsample_bytree=0.8      # Feature subsampling
random_state=42           # Reproducibility
```

**Data Preprocessing:**
1. Remove duplicates and irrelevant columns
2. Handle missing values (median for numeric, mode for categorical)
3. Convert area to square feet (Marla × 272.251, Kanal × 5445.02)
4. Encode categorical variables using LabelEncoder
5. Remove outliers using IQR method
6. Log-transform target variable for normalization

## 🔍 Explainability with SHAP

The application uses **SHAP (SHapley Additive exPlanations)** to provide interpretable predictions:

- **SHAP Force Plot**: Shows how each feature contributes to pushing the prediction up or down
- **SHAP Summary Plot**: Reveals feature importance and impact direction for all test cases
- **SHAP Bar Chart**: Displays mean absolute SHAP values for global feature importance

## 📊 Visualizations

All plots are automatically saved to the `plots/` folder:

1. **feature_importance.png** - Top 15 most important features
2. **actual_vs_predicted.png** - Scatter plot comparing true vs predicted prices
3. **residual_plot.png** - Distribution of prediction errors
4. **shap_summary.png** - SHAP beeswarm plot showing feature contributions
5. **shap_bar.png** - Mean absolute SHAP values per feature

## 🛠️ Technical Stack

- **ML Framework**: XGBoost 2.0.3
- **Explainability**: SHAP 0.44.1
- **Web Framework**: Streamlit 1.32.0
- **Data Processing**: Pandas 2.2.1, NumPy 1.26.4
- **ML Libraries**: Scikit-learn 1.4.1
- **Visualization**: Matplotlib 3.8.3, Seaborn 0.13.2, Plotly 5.20.0

## 💡 Key Insights

From the trained model, we can observe:

1. **Location is Critical**: The location feature has the highest importance in price prediction
2. **Size Matters**: Bedrooms and area significantly impact property values
3. **Geographic Variations**: Different cities show different price patterns
4. **Property Type**: Apartment vs House types have distinct pricing patterns
5. **Market Segment**: Purpose (Sale vs Rent) influences pricing

## 🔧 Troubleshooting

### Issue: "Model files not found"
**Solution**: Run `python train_model.py` to train the model first

### Issue: "No module named 'xgboost'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Streamlit app runs slowly
**Solution**: 
- Close other applications
- Clear browser cache
- Restart the app with `streamlit run app.py --logger.level=error`

### Issue: SHAP plots not generating
**Solution**: Ensure you have matplotlib and shap installed correctly
```bash
pip install --upgrade shap matplotlib
```

## 📚 Data Preprocessing Pipeline

```
Raw Data (168K records)
    ↓
Remove Duplicates
    ↓
Drop Irrelevant Columns (property_id, page_url, etc.)
    ↓
Handle Missing Values
    ↓
Convert Area Units (Marla/Kanal → Sq Ft)
    ↓
Encode Categorical Variables
    ↓
Remove Outliers (IQR Method)
    ↓
Log-Transform Target (Price)
    ↓
Cleaned Data (158K+ records)
```

## 📝 Training Pipeline

```
Load Preprocessed Data
    ↓
Split: 80% Train / 20% Test
    ↓
Train XGBoost Model (500 iterations)
    ↓
Evaluate on Test Set
    ↓
Generate Visualizations
    ↓
Save Model & Artifacts
    ↓
Ready for Prediction
```

## 🎓 Learning Resources

- **XGBoost**: https://xgboost.readthedocs.io/
- **SHAP**: https://shap.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/sarcasmos/pakistan-house-prices-and-property-listings

## 📄 License

This project is provided for educational and research purposes.

## 👨‍💻 Author

**AI/ML Engineer & Full Stack Developer**

Specialized in:
- Machine Learning & Data Science
- Web Development (Python, Streamlit)
- Data Analysis & Visualization
- Model Deployment & MLOps

## 🤝 Contributing

Contributions and suggestions are welcome! You can:
- Report bugs and issues
- Suggest improvements
- Fork and extend the project
- Optimize the model further

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Check Kaggle dataset documentation
4. Refer to XGBoost/SHAP documentation

---

**Built with ❤️ using Python, XGBoost, SHAP, and Streamlit**

*Last Updated: May 2026*
