# House Price Prediction — Regression from Fundamentals to Ridge

An end-to-end machine learning regression project built using Kaggle's
**House Prices: Advanced Regression Techniques** dataset.

This project explores the complete regression workflow, including:

- Exploratory Data Analysis
- Missing-value handling
- Train-test splitting
- Leakage-safe preprocessing
- One-hot encoding
- Feature scaling
- Linear Regression
- Linear Regression from scratch using NumPy and Gradient Descent
- Multicollinearity analysis
- Error analysis
- Ridge Regression for improved coefficient stability

The project was developed not only to predict house prices, but also to
understand how regression models learn coefficients, how preprocessing can
cause data leakage, and why regularization can improve model stability.

---

## Final Model Result

After correcting the preprocessing order so that preprocessing rules were
learned only from the training data, ordinary Linear Regression showed
instability caused by very large coefficients and produced several extreme
predictions.

To improve stability, I evaluated **Ridge Regression** using:

```python
Ridge(alpha=1.0)
```

On the 20% holdout test set, the leakage-safe Ridge model achieved:

| Metric | Ridge Regression |
|---|---:|
| MAE | 19,657.54 |
| RMSE | 36,356.11 |
| R² | 0.8277 |

The model explains approximately **82.8% of the variation in `SalePrice`**
on the holdout test set.

> R² = 0.8277 should not be interpreted as 82.8% classification accuracy.
> It represents the proportion of variation in the target explained by the
> regression model.

---

## Why I Built This

House-price prediction is a useful regression problem because the target is
continuous and depends on many numerical and categorical property features.

The main goals of this project were to understand:

1. how to build an end-to-end regression pipeline;
2. how Linear Regression learns coefficients;
3. how Gradient Descent works internally;
4. how data leakage affects evaluation;
5. how one-hot encoded features can create coefficient instability;
6. how Ridge Regression controls very large coefficients.

The project therefore combines both practical machine learning and
from-scratch implementation.

---

## Dataset

**Dataset:** Kaggle — House Prices: Advanced Regression Techniques

The training dataset contains:

| Property | Value |
|---|---:|
| Houses | 1,460 |
| Original columns | 81 |
| Target | `SalePrice` |

The repository also contains Kaggle's `test.csv`.

The metrics reported in this project are calculated using an internal
80/20 split of `train.csv`, not Kaggle's separate `test.csv`.

---

## Target Variable

The target variable is:

```text
SalePrice
```

The goal is to predict the selling price of a house from its property
characteristics.

---

## Example Features

The dataset contains numerical and categorical variables such as:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `YearBuilt`
- `Neighborhood`
- `MSZoning`
- `GarageType`
- `SaleCondition`

---

# Project Workflow

```text
train.csv
   │
   ▼
Dataset Inspection
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature / Target Separation
   │
   ├── Remove SalePrice
   └── Remove Id
   │
   ▼
Train-Test Split
   │
   │   80% Training
   │   20% Testing
   │
   ▼
Leakage-Safe Preprocessing
   │
   ├── Numerical Imputation
   │      └── Median learned from training data
   │
   ├── Categorical Imputation
   │      └── Missing values represented as "None"
   │
   ├── One-Hot Encoding
   │      └── Encoder fitted only on training data
   │
   └── Standard Scaling
          └── Mean/std learned only from training data
   │
   ▼
Modeling
   │
   ├── Ordinary Linear Regression
   │
   ├── NumPy Gradient Descent implementation
   │
   └── Ridge Regression
   │
   ▼
Evaluation
   │
   ├── MAE
   ├── MSE
   ├── RMSE
   ├── R²
   └── Per-house Error Analysis
```

---

# Exploratory Data Analysis

## Target Distribution

A histogram of `SalePrice` showed a **right-skewed distribution**.

Most houses were concentrated in the lower-to-middle price ranges, while a
smaller number of expensive houses formed a long right tail.

This is important because extreme high-price properties can have a strong
effect on regression errors.

---

## Numerical Relationships with SalePrice

Some of the strongest numerical correlations observed during EDA were:

| Feature | Correlation with `SalePrice` |
|---|---:|
| `OverallQual` | 0.7910 |
| `GrLivArea` | 0.7086 |
| `GarageCars` | 0.6404 |
| `GarageArea` | 0.6234 |
| `TotalBsmtSF` | 0.6136 |
| `1stFlrSF` | 0.6059 |
| `FullBath` | 0.5607 |
| `TotRmsAbvGrd` | 0.5337 |
| `YearBuilt` | 0.5229 |

`OverallQual` showed the strongest numerical correlation with `SalePrice`
among the inspected numerical features.

Correlation was used only as an exploratory relationship measure and not as
evidence of causation.

---

# Leakage-Safe Train-Test Split

Before performing learned preprocessing, the data was split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Resulting shapes:

```text
X_train: 1168 houses
X_test:   292 houses

y_train: 1168 values
y_test:   292 values
```

The key design decision was:

> Split first, preprocess second.

This prevents preprocessing statistics from the holdout test set from
influencing model training.

---

# Removing the Identifier

The `Id` column was removed from the corrected feature matrix:

```python
X = df_clean.drop(
    ["SalePrice", "Id"],
    axis=1
)

y = df_clean["SalePrice"]
```

`Id` is an identifier rather than a meaningful property characteristic.

---

# Missing-Value Handling

Missing values were handled after the train-test split.

## Numerical Features

Numerical missing values were filled using the median.

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(
    strategy="median"
)

X_train[numeric_cols] = num_imputer.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols] = num_imputer.transform(
    X_test[numeric_cols]
)
```

The important distinction is:

```text
Training data
→ fit_transform()
→ learn median + apply median

Test data
→ transform()
→ use training median only
```

This prevents leakage from the test set.

---

## Categorical Features

Missing categorical values were represented using:

```text
None
```

with:

```python
cat_imputer = SimpleImputer(
    strategy="constant",
    fill_value="None"
)
```

The imputer was fitted using the training data and then applied to the test
set.

---

# One-Hot Encoding

Machine-learning models cannot directly use text categories such as:

```text
GarageType = Attchd
Neighborhood = NAmes
```

Therefore categorical features were converted into binary dummy features
using `OneHotEncoder`.

Example:

```text
GarageType_Attchd   GarageType_Detchd
1                   0
0                   1
```

The encoder was fitted only on the training data:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat_encoded = encoder.fit_transform(
    X_train[categorical_cols]
)

X_test_cat_encoded = encoder.transform(
    X_test[categorical_cols]
)
```

`handle_unknown="ignore"` prevents errors if the test set contains a
category not observed during training.

---

# Feature Scaling

`StandardScaler` was used for models where feature scale matters,
especially Gradient Descent and Ridge Regression.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_encoded
)

X_test_scaled = scaler.transform(
    X_test_encoded
)
```

The scaler learns the training-data:

- mean;
- standard deviation.

The test set uses those same learned values.

---

# Multicollinearity Analysis

Linear Regression coefficients can become unstable when input features
contain strongly overlapping information.

I investigated this using:

1. feature-to-feature correlation;
2. Variance Inflation Factor.

---

## Feature-to-Feature Correlation

Several encoded features showed strong relationships.

Examples included:

- garage-related absence indicators;
- basement-related absence indicators;
- pool-related variables;
- sale-type and sale-condition indicators.

This demonstrated that different columns can sometimes carry very similar
information.

---

## Variance Inflation Factor

VIF was calculated for a selected set of important numerical features.

| Feature | VIF |
|---|---:|
| `GarageCars` | 5.23 |
| `GarageArea` | 4.97 |
| `GrLivArea` | 4.96 |
| `1stFlrSF` | 3.80 |
| `TotalBsmtSF` | 3.78 |
| `TotRmsAbvGrd` | 3.31 |
| `OverallQual` | 2.49 |
| `FullBath` | 2.17 |
| `YearBuilt` | 2.10 |

`GarageCars` had the highest VIF among this selected numerical feature set.

VIF was used as a diagnostic rather than automatically deleting every
feature above a fixed threshold.

---

# Model 1 — Ordinary Linear Regression

The project first evaluates Scikit-learn's Linear Regression:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(
    X_train_encoded,
    y_train
)

y_pred = model.predict(
    X_test_encoded
)
```

Linear Regression learns:

- one coefficient for each input feature;
- one intercept for the prediction equation.

Conceptually:

```text
Predicted Price
=
Intercept
+
Feature1 × Coefficient1
+
Feature2 × Coefficient2
+
...
```

---

# Linear Regression from Scratch

To understand what happens inside `.fit()`, Linear Regression was also
implemented manually using NumPy and Gradient Descent.

## Initialization

```python
weights = np.zeros(n_features)
bias = 0.0
```

---

## Prediction

```python
y_pred_train = X_train_np @ weights + bias
```

---

## Error

```python
errors = y_pred_train - y_train_np
```

---

## Mean Squared Error

```python
mse = np.mean(errors ** 2)
```

---

## Gradients

```python
dw = (2 / n_samples) * (
    X_train_np.T @ errors
)

db = (2 / n_samples) * np.sum(
    errors
)
```

---

## Parameter Updates

```python
weights = weights - learning_rate * dw
bias = bias - learning_rate * db
```

---

## Gradient Descent Flow

```text
Predict
   ↓
Calculate Error
   ↓
Calculate MSE
   ↓
Calculate Gradients
   ↓
Update Weights and Bias
   ↓
Repeat
```

This implementation helped connect the mathematical ideas of:

- weights;
- bias;
- cost;
- gradients;
- learning rate;
- iterations.

---

# Ordinary Linear Regression Stability Investigation

After rebuilding the pipeline so that preprocessing was performed correctly
after the train-test split, ordinary Linear Regression showed unstable
behavior.

Some one-hot encoded categorical features received extremely large
coefficients.

Examples included coefficients with magnitudes in the hundreds of thousands
or even millions.

This produced extreme predictions such as approximately:

```text
Minimum prediction: -840,308
Maximum prediction: 1,119,772
```

Examples of large errors included:

```text
Actual:     171,000
Predicted: -840,308

Actual:     241,500
Predicted: 1,119,772
```

This investigation demonstrated that a model can have severe coefficient
instability when:

- some categories are rare;
- encoded predictors are strongly related;
- ordinary least squares tries to fit many correlated features.

This motivated the use of regularization.

---

# Model 2 — Ridge Regression

Ridge Regression extends Linear Regression by adding a penalty for very
large coefficients.

Conceptually:

```text
Ridge Cost
=
Prediction Error
+
Penalty for Large Coefficients
```

This encourages the model to fit the data while keeping coefficients under
better control.

The model used:

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train_scaled,
    y_train
)

ridge_pred = ridge_model.predict(
    X_test_scaled
)
```

---

## What Does Alpha Mean?

`alpha` controls the strength of the Ridge penalty.

```text
Small alpha
→ weaker coefficient penalty

Large alpha
→ stronger coefficient penalty
```

This project uses:

```python
alpha = 1.0
```

as the evaluated configuration.

---

# Final Leakage-Safe Evaluation

The final Ridge model was evaluated on the 292-house holdout test set.

| Metric | Result |
|---|---:|
| MAE | 19,657.54 |
| MSE | 1,321,766,763.28 |
| RMSE | 36,356.11 |
| R² | 0.8277 |

---

## Metric Interpretation

### MAE

```text
MAE ≈ 19,658
```

MAE represents the average absolute prediction error across the holdout
houses.

---

### MSE

```text
MSE ≈ 1.322 billion
```

MSE squares prediction errors, so large mistakes receive greater influence.

---

### RMSE

```text
RMSE ≈ 36,356
```

RMSE is the square root of MSE and returns the error measure to the same
units as `SalePrice`.

Because errors are squared before averaging, RMSE is particularly sensitive
to large prediction mistakes.

---

### R²

```text
R² ≈ 0.8277
```

The Ridge model explains approximately **82.8% of the variation in
`SalePrice` on the holdout test set**.

This is not the same as saying the model has 82.8% accuracy.

---

# Error Analysis

Aggregate metrics alone do not reveal which houses are predicted badly.

For each holdout house, absolute error was calculated:

```python
ridge_error_check["Absolute_Error"] = abs(
    ridge_error_check["Actual"]
    - ridge_error_check["Predicted"]
)
```

The largest Ridge errors included:

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 241,500 | -82,383.58 | 323,883.58 |
| 171,000 | -72,090.86 | 243,090.86 |
| 755,000 | 580,384.74 | 174,615.26 |
| 611,657 | 470,525.95 | 141,131.05 |
| 556,581 | 424,854.56 | 131,726.44 |

The Ridge model greatly reduced the extreme predictions seen with ordinary
Linear Regression, but negative predictions were still possible.

This happens because ordinary linear regression models do not enforce:

```text
Predicted Price >= 0
```

---

# Why Ridge Helped

Ordinary Linear Regression allowed some coefficients to become extremely
large.

Ridge adds regularization:

```text
Linear Regression
+
Penalty for very large coefficients
```

This reduced coefficient instability and produced substantially more
reasonable predictions.

The comparison showed an important practical lesson:

> A more complex-looking model is not always required. Sometimes controlling
> model coefficients is enough to substantially improve stability.

---

# Visual Analysis

The notebook contains exploratory visualizations including:

- `SalePrice` distribution;
- feature-vs-target scatter plots;
- correlation analysis.

Standalone plot files have not yet been exported to the `plots/` directory.

---

# Repository Structure

```text
house-price-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── models/
│
├── plots/
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

The main implemented analysis currently lives in:

```text
notebooks/house_price_analysis.ipynb
```

The empty `models/`, `plots/`, and `src/` directories are reserved for
future project improvements.

---

# Requirements

The project uses:

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
jupyter
ipykernel
```

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/LAXMI15PRIYA/house-price-prediction.git
```

Then:

```bash
cd house-price-prediction
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 5. Open the Notebook

```text
notebooks/house_price_analysis.ipynb
```

Run the notebook cells in order.

> Note: On the development Windows environment used during this project,
> Windows Application Control blocked some SciPy compiled components.
> The corrected pipeline was therefore successfully executed in Google Colab.

---

# Key Engineering Lessons

## 1. Split Before Learned Preprocessing

The most important pipeline correction was:

```text
Wrong:
Preprocess entire dataset
→ split

Correct:
Split
→ fit preprocessing on training data
→ transform test data
```

This prevents data leakage.

---

## 2. Train Uses `fit_transform`, Test Uses `transform`

For imputers, encoders and scalers:

```text
Training:
fit_transform()

Testing:
transform()
```

The training set is allowed to teach preprocessing rules.

The test set should only receive those already-learned rules.

---

## 3. Error Analysis Matters

A single metric can hide very large individual mistakes.

Sorting predictions by absolute error exposed unrealistic predictions that
were not obvious from MAE alone.

---

## 4. RMSE Is Sensitive to Large Errors

MAE and RMSE behaved differently because RMSE gives greater influence to
large errors.

This helped identify extreme predictions.

---

## 5. Multicollinearity Can Destabilize Coefficients

Highly related input features can make individual Linear Regression
coefficients unstable.

Correlation analysis and VIF were useful diagnostics.

---

## 6. Rare Categories Can Cause Problems

Some one-hot encoded categories occur in very few houses.

A model has little data available to estimate their effects, which can lead
to unstable coefficients.

---

## 7. Ridge Regularization Improves Stability

Ridge Regression reduced the extreme coefficient behavior observed with
ordinary Linear Regression.

It provided a much more stable final regression baseline.

---

## 8. Scaling Is Important for Optimization and Regularization

Standardization was particularly important for:

- Gradient Descent;
- Ridge Regression.

Scaling ensures that features with larger numerical units do not dominate
optimization or regularization simply because of their scale.

---

# Limitations

The current project still has several limitations:

- Evaluation uses a single 80/20 holdout split rather than cross-validation.
- `SalePrice` is right-skewed and no log-target transformation is included
  in the final Ridge benchmark.
- Ridge Regression is still a linear model and cannot directly capture all
  nonlinear relationships.
- Ridge predictions are not constrained to positive values.
- Some rare categorical levels may still be difficult to estimate reliably.
- No systematic hyperparameter search for the optimal Ridge `alpha` has
  been performed.
- The Kaggle competition `test.csv` is not used for the reported evaluation
  metrics.
- No production API or deployed prediction application is included in this
  repository.

---

# Future Improvements

Possible next experiments include:

- cross-validation;
- Ridge `alpha` tuning;
- Lasso Regression;
- Elastic Net;
- log-transforming `SalePrice`;
- outlier investigation;
- feature engineering;
- rare-category grouping;
- Random Forest Regression;
- Gradient Boosting;
- XGBoost or LightGBM;
- reusable Scikit-learn `Pipeline` and `ColumnTransformer`;
- exporting trained models;
- adding standalone plots to the README.

---

# Skills Demonstrated

## Machine Learning

- Supervised learning
- Regression
- Linear Regression
- Ridge Regression
- Regularization
- Train-test evaluation

## Data Preparation

- Missing-value analysis
- Median imputation
- Categorical imputation
- One-hot encoding
- Standardization
- Leakage-safe preprocessing

## Statistical Analysis

- Correlation
- Multicollinearity
- Variance Inflation Factor

## Model Evaluation

- MAE
- MSE
- RMSE
- R²
- Per-observation error analysis

## Optimization

- Cost functions
- Gradient Descent
- Learning rate
- Gradients
- Weight updates
- Bias updates

## Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Jupyter Notebook
- Google Colab
- Git
- GitHub

---

# Main Takeaway

The most important outcome of this project was not simply obtaining an R²
score.

The project showed the full progression:

```text
Raw Data
   ↓
EDA
   ↓
Leakage-Safe Preprocessing
   ↓
One-Hot Encoding
   ↓
Scaling
   ↓
Ordinary Linear Regression
   ↓
Coefficient / Error Investigation
   ↓
Ridge Regression
   ↓
More Stable Final Model
```

The biggest learning was that a machine-learning project is not only about
training a model.

A reliable project also requires:

- correct preprocessing order;
- prevention of data leakage;
- meaningful evaluation;
- coefficient analysis;
- error analysis;
- understanding model limitations;
- improving the model based on evidence.