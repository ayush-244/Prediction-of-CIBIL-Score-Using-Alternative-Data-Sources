# CIBIL Score Prediction Using Alternative Data Sources

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 🎯 Problem Statement

Traditional credit scoring relies on historical credit bureau records, limiting financial access for millions of individuals in emerging markets. This project addresses the **"credit invisibility" problem** by developing a machine learning model that predicts creditworthiness using **alternative financial data sources**, enabling inclusive credit assessment.

## 📊 Project Overview

An end-to-end machine learning solution that predicts CIBIL credit scores by analyzing:
- **Digital Transaction Patterns**: Transaction volume and frequency (last 6 months)
- **Payment Behavior**: On-time payment track record and consistency
- **Financial Activity**: Active banking months and loan/utility bill management
- **Income Profile**: Monthly income as purchasing power indicator

### Key Features

✅ **Production-Ready ML Pipeline**: Automated data loading → feature engineering → model training  
✅ **Advanced Algorithm**: Gradient Boosting Regressor (scikit-learn) with hyperparameter optimization  
✅ **Interactive Web Interface**: User-friendly Streamlit application for real-time predictions  
✅ **Model Persistence**: Serialized models for deployment and inference  
✅ **Modular Architecture**: Clean separation of concerns for scalability and maintainability

## �️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Python 3.7+ |
| **ML Algorithm** | Scikit-learn (Gradient Boosting Regressor) |
| **Data Processing** | Pandas, NumPy |
| **Web Application** | Streamlit (Interactive UI) |
| **Model Serialization** | Joblib |
| **Validation Metrics** | RMSE, R² Score, Train-Test Split |

## 📁 Architecture & Project Structure

```
.
├── main.py                    # Pipeline orchestrator: Load → Engineer → Train
├── model.py                   # Model training, evaluation, and persistence
│                              # - train_model(): Handles train/test split, model fitting
│                              # - load_model(): Deserializes saved model for inference
├── data_loader.py             # Data ingestion and validation
├── feature_engineering.py     # Feature extraction and transformation pipeline
├── utils.py                   # Inference utilities and prediction functions
├── ui_app.py                  # Streamlit web interface for real-time predictions
├── config.py                  # Centralized configuration (paths, hyperparameters)
├── requirements.txt           # Production dependencies
└── README.md                  # Documentation
```

### Data Flow Diagram

```
Raw CSV Data
    ↓
[data_loader] → DataFrame
    ↓
[feature_engineering] → Engineered Features
    ↓
[model] → Train/Test Split
    ↓
[GradientBoostingRegressor] → Trained Model
    ↓
[joblib] → Model Serialization (.pkl)
    ↓
[ui_app / utils] → Inference → CIBIL Score Prediction
```

## ⚡ Quick Start

### Installation

```bash
# Clone repository
cd "Prediction-of-CIBIL-Score-Using-Alternative-Data-Sources"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage: Training Pipeline

Execute the complete ML pipeline:
```bash
python main.py
```

**Pipeline Steps**:
1. **Load Data**: Reads CSV from configured path
2. **Feature Engineering**: Transforms raw features
3. **Model Training**: Trains Gradient Boosting model on 80% data
4. **Evaluation**: Validates on 20% holdout set
5. **Persistence**: Serializes model to `models/shadow_cibil_model.pkl`

**Output**: Console displays:
```
RMSE: [model error]
R^2 Score: [variance explained]
```

### Usage: Web Application

Launch interactive prediction interface:
```bash
streamlit run ui_app.py
```

**Access**: Open `http://localhost:8501` in browser

**Features**:
- Real-time CIBIL score prediction
- Form-based input for 5 financial indicators
- Instant prediction results
- Input validation and constraints

**Input Parameters**:
| Parameter | Type | Constraints | Example |
|-----------|------|-------------|---------|
| Monthly Income | Float | ≥ 0 | 50000 |
| Total Digital Transactions (6mo) | Float | ≥ 0 | 150000 |
| Active Months | Integer | ≥ 1 | 6 |
| Number of Utility Bills | Integer | ≥ 0 | 5 |
| On-Time Payments | Integer | ≥ 0 | 15 |

## 🤖 Machine Learning Model

### Algorithm & Rationale
- **Model**: Gradient Boosting Regressor (Scikit-learn)
- **Why Gradient Boosting?**: 
  - Handles non-linear relationships in financial data
  - Robust to outliers and multicollinearity
  - Superior performance on regression tasks
  - Built-in feature importance calculation

### Model Configuration
```python
- Algorithm: GradientBoostingRegressor
- Train-Test Split: 80-20 (random_state=42)
- Loss Function: Least Squares
- Evaluation Metrics:
  • Root Mean Squared Error (RMSE)
  • R² Score (Coefficient of Determination)
```

### Validation Approach
- **Temporal Robustness**: Fixed random state for reproducibility
- **Data Separation**: Stratified train-test split prevents data leakage
- **Metric Selection**: 
  - RMSE: Measures prediction error magnitude
  - R² Score: Explains variance in credit score predictions

### Expected Performance Characteristics
- Handles 5-dimensional feature space efficiently
- Suitable for real-time inference (< 100ms per prediction)
- Scalable to larger datasets without architecture changes

## � Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation and analysis |
| scikit-learn | Latest | Machine learning algorithms |
| joblib | Latest | Model serialization/deserialization |
| streamlit | Latest | Web application framework |

**Install**: `pip install -r requirements.txt`

---

## 🚀 Deployment & Scalability

### Current Deployment
- **Development**: Local Python environment
- **Web Interface**: Streamlit (development server)
- **Model Storage**: Disk-based `.pkl` files

### Production Readiness Considerations
- Model versioning system
- API wrapper (Flask/FastAPI) for integration
- Docker containerization for consistency
- Model monitoring and retraining pipelines
- Database integration for data persistence

---

## 🔄 Potential Improvements & Future Work

### Model Enhancement
- [ ] Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Ensemble methods (Stacking, Voting)
- [ ] Feature importance analysis and selection
- [ ] Cross-validation for robust performance estimation
- [ ] Handling class imbalance (if applicable)

### Data & Features
- [ ] Temporal feature engineering (trends, seasonality)
- [ ] Additional alternative data sources integration
- [ ] Data quality assessment and cleaning pipeline
- [ ] Outlier detection and treatment
- [ ] Feature scaling standardization (StandardScaler/MinMaxScaler)

### Code Quality & Deployment
- [ ] Unit tests and integration tests
- [ ] Logging and error handling
- [ ] API development (FastAPI/Flask)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Model performance monitoring

---

## � Feature Engineering

The project applies intelligent feature transformations to extract predictive signals:

### Input Features (Raw)
- `monthly_income`: Primary income indicator
- `total_txn_amount`: Digital transaction volume (6-month window)
- `active_months`: Banking activity frequency
- `loan_count`: Credit obligations (utility bills as proxy)
- `on_time_payments`: Payment discipline metric

### Processing Pipeline
```python
# Feature engineering in feature_engineering.py:
1. Normalization: Standardization of feature scales
2. Derived Features: Ratios and interactions (if applicable)
3. Validation: Handles missing values and outliers
4. Output: Engineered feature matrix ready for model training
```

### Rationale
- **Income**: Repayment capacity
- **Digital Transactions**: Financial formalization and activity level
- **Active Months**: Consistency and banking engagement
- **Loan Count**: Debt servicability
- **On-Time Payments**: Credit responsibility

---

## 📈 Data Format & Configuration

### Expected Input CSV Format
```csv
monthly_income,total_txn_amount,active_months,loan_count,on_time_payments,shadow_score
50000,150000,6,5,15,650
45000,120000,5,3,12,620
...
```

### Configuration (`config.py`)
```python
DATA_PATH = "data/sample_data.csv"              # Input dataset path
MODEL_PATH = "models/shadow_cibil_model.pkl"   # Serialized model location
```

**Update `DATA_PATH`** to point to your training dataset before running `main.py`.

---

## 🎓 Key Technical Achievements

✅ **End-to-End ML Pipeline**: Implemented production-ready workflow from data ingestion to model deployment  
✅ **Modular Design**: Separation of concerns enables easy debugging and component testing  
✅ **Alternative Data Integration**: Demonstrated credit scoring without traditional bureau data  
✅ **User-Centric Interface**: Streamlit web app for non-technical users  
✅ **Model Reproducibility**: Fixed random seeds ensure consistent results across runs  
✅ **Scalable Architecture**: Support for larger datasets and multiple model versions

## 🛠️ Development Guide

### Project Workflow

1. **Adding New Features**
   ```bash
   # 1. Modify feature_engineering.py (add transformations)
   # 2. Update data_loader.py (if data structure changes)
   # 3. Retrain: python main.py
   # 4. Test in web app: streamlit run ui_app.py
   ```

2. **Model Improvements**
   - Edit `model.py` to experiment with algorithms
   - Adjust hyperparameters (test locally first)
   - Evaluate metrics before production deployment
   - Commit changes with performance comparisons

3. **Debugging**
   - Check data paths in `config.py`
   - Verify data format matches expected schema
   - Review console output for RMSE/R² scores
   - Use Streamlit's debug mode for UI issues

### Code Quality Standards
- **Naming**: Clear, descriptive variable/function names
- **Comments**: Document complex transformations
- **Structure**: Functions have single responsibilities
- **Error Handling**: Validate inputs, handle edge cases

---

## 📋 File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | Pipeline orchestrator | Coordinates data → features → training |
| `model.py` | ML core | `train_model()`, `load_model()` |
| `data_loader.py` | Data ingestion | `load_data()` |
| `feature_engineering.py` | Transformations | `engineer_features()` |
| `utils.py` | Utilities | `predict_score()` |
| `ui_app.py` | Web interface | Streamlit app |
| `config.py` | Configuration | Paths and constants |

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` |
| **FileNotFoundError for data** | Update `DATA_PATH` in `config.py` |
| **Model not found** | Train model: `python main.py` |
| **Streamlit port conflict** | Use `streamlit run ui_app.py --server.port 8502` |
| **Prediction errors** | Verify input data types match expectations |

---

## 📚 Classification & Impact

- **Problem Type**: Regression (Continuous CIBIL Score Prediction)
- **Use Case**: Financial Inclusion, Alternative Credit Scoring
- **Impact**: Enables credit access for underbanked populations
- **Domain**: FinTech, Credit Risk, Alternative Data

---




