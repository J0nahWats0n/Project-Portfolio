'''
Jonah Watson
Spring 2026
Mental and Behavioral Consequences of Alcohol Consumption Among College Students
This file was used to test an ordinal logistic regression model predicting risk of alcohol misuse
'''


from sklearn.model_selection import train_test_split   
# imported to split data into training and testing sets

from sklearn.preprocessing import StandardScaler
# imported to standardize continuous variables

from sklearn.pipeline import Pipeline
# imported to create modeling pipelines

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# imported to evaluate model performance

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# imported to create and display confusion matrices

from statsmodels.stats.outliers_influence import variance_inflation_factor
# imported to check for multicollinearity among variables

import statsmodels.api as sm
# imported for statistical modeling

from statsmodels.miscmodels.ordinal_model import OrderedModel
# imported to perform ordinal logistic regression

import matplotlib.pyplot as plt   # imported for data visualization
import numpy as np                # imported for numerical operations
import pandas as pd               # imported for data manipulation and analysis
import seaborn as sns             # imported for enhanced data visualization



# ================
# Data Preparation
# ================

# load the cleaned dataset
df = pd.read_csv("CLEANED ALCOHOL NCHA-III S24 - New_Numeric.csv")

# performing answer recoding for consistency 
binary_vars = [
    "N3Q65A2", "N3Q65A3", "N3Q65A7", "N3Q65A15",
    "N3Q65A19", "N3Q65A28", "N3Q65A31",
    "N3Q65A33", "N3Q65A35",
    "N3Q77A", "N3Q77B",
    "RULS3", "RSBQR"   
]
# "N3Q25B1", "N3Q25B2", "N3Q29A", "N3Q29B", "N3Q29C", "N3Q29D", "N3Q29E", 
# "N3Q29F", "N3Q29G", "N3Q29H", "N3Q29I", "N3Q29J", "N3Q29K", "N3Q29L", "N3Q30A",

for col in binary_vars: 
    df[col] = df[col].replace({1: 0, 2: 1})

# standardizing continuous scale variables so they have mean=0 and std=1
scale_vars = [
    "DIENER",
    "CDRISC2"
]

scaler = StandardScaler()
df[scale_vars] = scaler.fit_transform(df[scale_vars])

# view how much each standard deviation unit corresponds to in original units
diener_sd = scaler.scale_[scale_vars.index("DIENER")]
print(f"DIENER standard deviation weight: {diener_sd}")
cdrisc2_sd = scaler.scale_[scale_vars.index("CDRISC2")]
print(f"CDRISC2 standard deviation weight: {cdrisc2_sd}")

# special case
df["RKESSLER6"] = df["RKESSLER6"].replace({1: 0, 3: 1})

# drop NA from variables that will be used in modeling
model_variables = binary_vars + scale_vars + ["N3Q1", "N3Q48", "N3Q80", "RKESSLER6", "ALCOHOLRISK"]
df = df.dropna(subset=model_variables)

# flip overall health so higher is better
df['N3Q1'] = 6 - df['N3Q1']     

# flip GPA so higher is better
df["N3Q80"] = df["N3Q80"].max() - df["N3Q80"] + 1



# choose which variables will be included in the modeling process
X = df[["N3Q1",      # Overall health
    "N3Q48",         # Stress
    "N3Q77A",        # Greek membership
    "N3Q77B",        # Greek housing
    "N3Q80",         # GPA

    # mental health diagnoses
    "N3Q65A2",       # ADHD
    "N3Q65A7",       # Anxiety
    "N3Q65A15",      # Depression
    "N3Q65A28",      # Insomnia
    "N3Q65A31",      # OCD
    "N3Q65A33",      # PTSD
    "N3Q65A35",      # Sleep apnea

    # mental-Health Scales
    "RKESSLER6",     # Psychological distress scale
    "RULS3",         # Loneliness scale
    "RSBQR",         # Suicide risk scale
    "DIENER",        # Well-being scale
    "CDRISC2"        # Resilience scale
]]

# remap target variable
y = df["ALCOHOLRISK"].map({
    1: 0,
    2: 1,
    3: 2,
})



# ==============================
# COLLINEARITY CHECKS
'''check for collinearity among groups of variables to ensure no variables are 
too highly correlated to be included together in the model'''
# ==============================



# mental health diagnoses
# the highest value was 0.65, which is good, as above 0.8 indicates high correlation 
# which is undesirable for modeling
diagnoses = ["N3Q65A2", "N3Q65A3", "N3Q65A7", "N3Q65A15", "N3Q65A19", "N3Q65A28", 
             "N3Q65A31", "N3Q65A33", "N3Q65A35"]
correlation_matrix = df[diagnoses].corr()
print(correlation_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Mental Health Diagnoses")
plt.savefig("mental_health_diagnoses_correlation_matrix.png")

# Variance Influence Factor (VIF) calculation to ensure no multicollinearity among mental health scales
# the highest value was 2.65, which is good, as above 5 indicates high multicollinearity 
# which is undesirable for modeling
X_diagnoses = df[diagnoses]
vif_data = pd.DataFrame()
vif_data["feature"] = X_diagnoses.columns
vif_data["VIF"] = [variance_inflation_factor(X_diagnoses.values, i) for i in range(X_diagnoses.shape[1])]
print(vif_data)



# mental health scales
# the highest value was 0.53, which is good, as above 0.8 indicates high correlation
# which is undesirable for modeling
scales = ["RKESSLER6", "RULS3", "DIENER", "RSBQR", "CDRISC2"]
correlation_matrix = df[scales].corr()
print(correlation_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Mental Health Scales")
plt.savefig("mental_health_scales_correlation_matrix.png")

# Variance Influence Factor (VIF) calculation to ensure no multicollinearity among mental health scales
# the highest value was 1.64, which is good, as above 5 indicates high multicollinearity
# which is undesirable for modeling
X_scales = df[scales]
vif_data = pd.DataFrame()
vif_data["feature"] = X_scales.columns
vif_data["VIF"] = [variance_inflation_factor(X_scales.values, i) for i in range(X_scales.shape[1])]
print(vif_data)



# ==============================
# Modeling
# ==============================



# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=1
)

# scale features to make their mean=0 and std=1
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ordinal logistic regression model
ordinal_model = OrderedModel(
    y_train,
    X_train_scaled,
    distr="logit"   # logistic link which is standard for ordinal logistic regression
)

ordinal_results = ordinal_model.fit(method="bfgs")

print(ordinal_results.summary())

# Class probabilities
y_pred_prob = ordinal_results.model.predict(
    ordinal_results.params,
    exog=X_test_scaled
)

# Expected ordinal value (better for ordinal outcomes)
classes = np.arange(y_pred_prob.shape[1])
y_pred_expected = np.dot(y_pred_prob, classes)

# evaluate ordinal distance (mean absolute error)
mae = np.mean(np.abs(y_test - y_pred_expected))
print(f"Mean Absolute Error (ordinal): {mae:.3f}")

# evaluate within +/- 1 category
within_one = np.mean(np.abs(y_test - y_pred_expected) <= 1)
print(f"Within +/- 1 category: {within_one:.3f}")

# predicted class = max probability
y_pred = np.argmax(y_pred_prob, axis=1)

'''
AI assistance was used to suggest solutions and resolve errors
'''