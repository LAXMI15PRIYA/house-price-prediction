# House Price Prediction — Linear Regression from Scikit-learn to NumPy

An end-to-end regression project built on Kaggle's **House Prices: Advanced Regression Techniques** dataset.

The project goes beyond calling `LinearRegression().fit()`: it implements the same prediction idea manually with **NumPy and Gradient Descent**, then compares that implementation against Scikit-learn on the same holdout split.

On the current 20% test split, the Scikit-learn model achieved **R² = 0.8557** with **RMSE = 33,269**, while the NumPy Gradient Descent implementation achieved **R² = 0.8535** with **RMSE = 33,526**.

The close results provide a practical check that the from-scratch implementation learned a similar regression relationship.

---

## Why I Built This

House-price prediction is a useful regression problem because the target is continuous and depends on many interacting property characteristics.

For this project, I wanted to understand both:

- how to build a complete regression workflow using standard ML tools;
- what actually happens inside Linear Regression when coefficients are learned.

The project therefore contains two implementations:

1. **Scikit-learn Linear Regression**
2. **Linear Regression implemented manually with NumPy and Gradient Descent**

The goal is educational and analytical rather than to build a real-world property valuation service.

---

## What I Implemented

The notebook covers the complete workflow used in this project:

- Dataset inspection with Pandas
- Target-distribution analysis
- Scatter plots and correlation analysis
- Missing-value investigation
- Categorical and numerical missing-value handling
- One-hot encoding
- 80/20 train-test split
- Standardization with `StandardScaler`
- Feature-to-feature correlation analysis
- Multicollinearity investigation
- Variance Inflation Factor (VIF)
- Scikit-learn Linear Regression
- MAE, MSE, RMSE and R² evaluation
- Coefficient and intercept inspection
- Linear Regression implemented with NumPy
- Gradient Descent
- Manual weight and bias updates
- Scikit-learn vs scratch-model comparison
- Per-house prediction error analysis

---

## Dataset

**Source:** Kaggle — House Prices: Advanced Regression Techniques

The provided training dataset contains:

| Property | Value |
|---|---:|
| Training records | 1,460 houses |
| Original columns | 81 |
| Target | `SalePrice` |

The repository also contains Kaggle's `test.csv`, but the metrics reported in this README are **not** calculated on that file.

For model evaluation, `train.csv` was divided into an internal training and holdout test set.

### Target

```text
SalePrice
```

### Example predictors

The dataset contains numerical and categorical information such as:

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

After one-hot encoding, the model matrix contained **260 features**.

---

## Workflow

```text
train.csv
   │
   ├── Dataset inspection
   │
   ├── Exploratory Data Analysis
   │
   ├── Missing-value handling
   │
   ├── Feature / target separation
   │
   ├── One-hot encoding
   │
   ├── 80/20 train-test split
   │
   ├── Multicollinearity analysis
   │
   ├── Scikit-learn Linear Regression
   │        │
   │        └── Holdout evaluation
   │
   └── StandardScaler
            │
            └── NumPy Linear Regression
                 │
                 ├── Gradient Descent
                 └── Holdout evaluation
```

---

## Exploratory Data Analysis

### Target distribution

A histogram of `SalePrice` showed a **right-skewed distribution**: most houses were concentrated in the lower-to-middle price ranges, with a smaller number of high-priced properties forming a long right tail.

### Numerical relationships with SalePrice

Some of the strongest numerical correlations observed were:

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

`OverallQual` had the strongest numerical correlation with the target among the features inspected.

Correlation was used as an exploratory tool, not as evidence that a feature directly causes a price change.

---

## Missing-Value Handling

Missing values were first counted with:

```python
df.isnull().sum()
```

The treatment depended on what a missing value represented.

### Categorical absence

For several categorical variables, missing values represented the absence of a property feature.

Examples included garage-, basement-, pool-, fireplace- and fence-related variables.

Those categories were represented using:

```text
None
```

rather than treating every missing category as a generic data error.

### Numerical values

The current notebook uses median imputation for:

- `LotFrontage`
- `MasVnrArea`
- `GarageYrBlt`

The remaining missing value in `Electrical` was filled using the mode.

### Important evaluation caveat

In the current notebook, some imputation values were calculated **before the train-test split**.

That allows information from the eventual holdout set to influence preprocessing statistics.

A stricter version of the pipeline should:

1. split the data first;
2. fit imputers on `X_train`;
3. transform `X_train` and `X_test` using those training-derived values.

This is listed as a next improvement rather than being hidden from the reported results.

---

## Categorical Encoding

Categorical variables were converted into model-compatible numerical features using:

```python
X_encoded = pd.get_dummies(X, drop_first=True)
```

`drop_first=True` removes one dummy level from each encoded categorical variable.

After encoding:

```text
Rows:      1,460
Features:    260
```

One-hot encoding was performed before the current train-test split. A stricter reusable pipeline would learn categorical levels from training data and apply the same encoder to the holdout set.

---

## Train-Test Split

The encoded dataset was divided using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)
```

Resulting shapes:

```text
X_train: (1168, 260)
X_test:   (292, 260)

y_train:  (1168,)
y_test:    (292,)
```

The 292 test houses were not used to fit either regression model.

---

## Feature Scaling

The NumPy Gradient Descent implementation uses standardized features:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This part of the preprocessing is leakage-safe:

- the scaler learns its mean and standard deviation from `X_train`;
- `X_test` is transformed using those same training-derived values.

Scaling is especially useful for the Gradient Descent implementation because the original predictors have very different numerical ranges.

The Scikit-learn `LinearRegression` model was trained on the unscaled encoded features.

---

## Multicollinearity Analysis

Linear Regression coefficients can become difficult to interpret when predictors carry strongly overlapping information.

I investigated this in two ways.

### Feature-to-feature correlation

The encoded feature correlation matrix was filtered to inspect pairs with high absolute correlation.

Examples included relationships among:

- garage-absence indicator columns;
- basement-absence indicator columns;
- pool-related variables;
- sale-type and sale-condition indicators.

This highlighted cases where different columns represented very similar underlying information.

### Variance Inflation Factor

VIF was calculated for a selected group of important numerical predictors.

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

`GarageCars` had the largest VIF in this selected feature set.

I treated VIF as a diagnostic rather than automatically deleting any feature above a fixed threshold.

---

## Model 1 — Scikit-learn Linear Regression

The first implementation used Scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

`LinearRegression` solves the ordinary least-squares problem using numerical linear algebra rather than the Gradient Descent loop used in the scratch implementation.

The fitted model learned:

- one coefficient for every encoded input feature;
- one intercept for the regression equation.

---

## Model 2 — Linear Regression from Scratch

The second implementation was written with NumPy to understand the optimization process directly.

### Initialization

```python
weights = np.zeros(n_features)
bias = 0.0
```

With 260 features, the model starts with 260 weights and one bias term.

### Prediction

```python
y_pred_train = X_train_np @ weights + bias
```

This performs the weighted combination of all input features and adds the bias.

### Error

```python
errors = y_pred_train - y_train_np
```

### Mean Squared Error

```python
mse = np.mean(errors ** 2)
```

### Gradients

```python
dw = (2 / n_samples) * (X_train_np.T @ errors)
db = (2 / n_samples) * np.sum(errors)
```

### Parameter updates

```python
weights = weights - learning_rate * dw
bias = bias - learning_rate * db
```

### Training configuration

```python
learning_rate = 0.01
iterations = 2000
```

The complete optimization loop repeatedly:

```text
Predict
   ↓
Measure error
   ↓
Calculate gradients
   ↓
Update weights and bias
   ↓
Repeat
```

The final scratch model learned **260 weights** and a bias of approximately **181,441.54**.

---

## Evaluation

Both models were evaluated on the same **292-house holdout set**.

### Results

| Metric | Scikit-learn | NumPy + Gradient Descent |
|---|---:|---:|
| MAE | 20,291.08 | **19,830.21** |
| MSE | **1,106,826,200.91** | 1,123,991,638.15 |
| RMSE | **33,269.00** | 33,525.98 |
| R² | **0.8557** | 0.8535 |

### Interpretation

The Scikit-learn model achieved:

```text
R² = 0.8557
```

meaning it explained approximately **85.6% of the variation in `SalePrice` within the current holdout set**.

The NumPy implementation achieved:

```text
R² = 0.8535
```

with an RMSE only slightly higher than the Scikit-learn model.

The small difference is expected because the models were fitted differently:

```text
Scikit-learn
    → numerical least-squares solver

NumPy implementation
    → iterative Gradient Descent
```

The similarity of the metrics is evidence that the manual optimization implementation learned a comparable regression relationship.

These values should be treated as the results of the **current notebook pipeline**, not as a final benchmark, because the preprocessing-order leakage described above should be corrected before making a stronger evaluation claim.

---

## Error Analysis

Aggregate metrics do not show which individual predictions fail badly, so I also calculated the absolute prediction error for each test house:

```python
error_analysis["Error"] = abs(
    error_analysis["Actual"] - error_analysis["Predicted"]
)
```

and sorted the largest errors first.

Examples from the scratch model included:

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 171,000 | -58,027.62 | 229,027.62 |
| 241,500 | 43,335.12 | 198,164.88 |
| 755,000 | 584,848.94 | 170,151.06 |
| 611,657 | 456,864.71 | 154,792.29 |

This analysis exposed an important limitation of ordinary Linear Regression:

> predictions are not constrained to be positive.

The largest-error cases also provide useful starting points for investigating:

- high-priced or unusual properties;
- outliers;
- nonlinear relationships;
- missing predictive information;
- model assumptions that may not hold uniformly across the dataset.

---

## Visual Analysis

The notebook contains the EDA visualizations used during analysis, including the `SalePrice` distribution and feature-vs-target scatter analysis.

No standalone image filenames inside `plots/` were provided with the reviewed material, so this README intentionally does **not** reference unverified image paths.

A future repository cleanup can export the most useful figures to `plots/` and embed them here.

---

## Repository Structure

Based on the current project structure:

```text
house-price-prediction/
│
├── .venv/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── plots/
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

`.venv/` is a local development environment and should remain excluded from Git.

The primary implemented analysis currently lives in:

```text
notebooks/house_price_analysis.ipynb
```

The presence of `src/`, `models/`, and `plots/` does not imply that production modules, serialized models, or exported figures already exist there.

---

## Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

The repository contains `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### 5. Open the notebook

```text
notebooks/house_price_analysis.ipynb
```

Run the notebook cells from top to bottom so that preprocessing variables, fitted models and evaluation outputs are created in the correct order.

---

## Engineering Lessons

### 1. A good metric does not mean every prediction is good

An R² above 0.85 still coexisted with individual prediction errors above 150,000 price units.

This is why I added per-house error analysis instead of relying only on aggregate metrics.

### 2. Multicollinearity matters most when interpreting coefficients

Highly related predictors can still support useful predictions, but they make individual coefficient effects harder to interpret reliably.

Correlation and VIF helped identify this issue.

### 3. Feature scaling matters for optimization

Ordinary Scikit-learn Linear Regression did not require scaling for the solver used here, but the manually implemented Gradient Descent model benefited from standardized feature scales.

### 4. Gradient Descent made the model mechanics concrete

Implementing the algorithm manually connected:

```text
features
→ weights
→ predictions
→ errors
→ MSE
→ gradients
→ parameter updates
```

rather than treating `.fit()` as a black box.

### 5. Evaluation pipelines need careful preprocessing order

The current notebook correctly fits `StandardScaler` only on training data, but some earlier preprocessing was performed before the split.

That distinction is important because even subtle access to holdout information can make evaluation less clean.

---

## Limitations

The current implementation has several known limitations:

- Median/mode imputation is performed before the train-test split.
- One-hot encoding is created before the split.
- `Id` is currently retained as a predictor even though it is an identifier.
- Evaluation uses a single 80/20 split rather than cross-validation.
- The target distribution is right-skewed, but no target transformation was evaluated.
- Ordinary Linear Regression cannot model complex nonlinear relationships directly.
- Predictions are not constrained to valid positive house-price values.
- The Kaggle `test.csv` file is present but is not used for the metrics reported here.
- No tuned regularized or nonlinear model has yet been evaluated in this project.

These limitations are intentionally documented instead of being hidden behind the headline metric.

---

## Next Steps

### Evaluation cleanup

- Split the data before fitting any learned preprocessing.
- Fit imputation values using training data only.
- Fit categorical encoding using training data only.
- Remove `Id` from model features.
- Re-run the holdout evaluation after those corrections.

### Modeling experiments

After establishing the corrected baseline:

- log-transform `SalePrice` and evaluate the effect;
- use cross-validation;
- compare Ridge and Lasso regression;
- investigate outlier handling;
- engineer additional meaningful features;
- compare against nonlinear regression models.

### Repository improvements

- Move reusable preprocessing/model code from the notebook into `src/`.
- Export the strongest EDA and model-diagnostic plots to `plots/`.
- Add saved model artifacts only if model persistence is actually implemented.
- Pin dependency versions in `requirements.txt` if reproducibility across environments becomes important.

---

## Skills Demonstrated

This project demonstrates hands-on understanding of:

**Machine Learning**
- Supervised learning
- Regression
- Train-test evaluation
- Model interpretation

**Data Preparation**
- Missing-value analysis
- Numerical and categorical imputation
- One-hot encoding
- Standardization

**Statistical Analysis**
- Correlation
- Multicollinearity
- Variance Inflation Factor

**Model Evaluation**
- MAE
- MSE
- RMSE
- R²
- Per-observation error analysis

**Optimization**
- Cost functions
- Gradients
- Gradient Descent
- Learning rate
- Iterative parameter updates

**Implementation**
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Jupyter Notebook

---

## Main Takeaway

The most valuable part of this project was not simply achieving an R² around 0.85.

It was implementing the regression learning process manually and connecting the mathematical concepts to working code:

```text
Linear Regression
        │
        ├── Scikit-learn least-squares implementation
        │
        └── NumPy Gradient Descent implementation
                       │
                       └── comparable holdout performance
```

That comparison turned Linear Regression from a library call into an algorithm I can explain, implement and debug step by step.