'''
Jonah Watson
Fall 2025
Sleep Health in College Students: A Multivariable Predictive Modeling Analysis
This file was the second iteration of modeling practices, this time testing a logistic regression model, lasso, and decision tree.
'''

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
# imported for evaluation metrics and access to the confusion matrix and its display

from sklearn.tree import DecisionTreeClassifier             # imported for integration of decision tree
from sklearn.linear_model import LogisticRegression         # imported for logistic regression modeling
from sklearn.pipeline import make_pipeline                  # imported for creating a machine learning pipeline
from sklearn.ensemble import RandomForestClassifier         # imported for creating a random forest model
from sklearn.impute import SimpleImputer                    # imported for handling missing data
from sklearn.preprocessing import StandardScaler            # imported for feature scaling
from sklearn.model_selection import train_test_split        # imported for splitting data into training and testing sets
import matplotlib.pyplot as plt                             # imported for visualizing confusion matrix
import numpy as np                                          # imported for numerical operations
import pandas as pd                                         # imported for data manipulation
import statsmodels.api as sm                                # imported for statistical modeling



# load the cleaned dataset
df = pd.read_csv("CLEANED SLEEP NCHA-III S24 - New_Numeric.csv")



# choose which variables will be included in the modeling process
df_filtered = df[['N3Q1', 'N3Q14', 'N3Q42B', 'N3Q48', 'N3Q20D', 'N3Q20F', 'N3Q20G', 'N3Q65A2', 'N3Q65A3', 
               'N3Q65A7', 'N3Q65A15', 'N3Q65A28', 'N3Q65A35', 'RULS3']]

# recode the target variable (Weeknight sleep duration) as this model does not include students receiving greater than 9 hours of sleep per night
df_filtered = df[df['N3Q14'] != 8]          



print(df_filtered['N3Q14'].value_counts())
df_filtered['sleep_binary'] = df_filtered['N3Q14'].apply(lambda x: 1 if x >= 5 else 0)  # lambda function to create binary target variable
# 1 represents if a student averages 7-9 hours of sleep per night (optimal)
# 0 represents if a student obtains 6 hours of sleep or less per night (suboptimal)
print(df_filtered['sleep_binary'].value_counts())

# recode binary variables so that their responses switch from 1s and 2s to 0s and 1s, where 1 will indicate the presence of the condition
binary_vars = ['N3Q20F', 'N3Q20G', 'N3Q65A2', 'N3Q65A3',    
               'N3Q65A7', 'N3Q65A15', 'N3Q65A28', 'N3Q65A35', 'RULS3']  
for col in binary_vars:
    df_filtered[col] = df_filtered[col].replace({1: 0, 2: 1}) 
    
# recode this question so that higher values indicate better overall health to be consistent for the format of other questions
df_filtered['N3Q1'] = 6 - df_filtered['N3Q1']       



"""The next few lines of code separate the predictors (X) and the target variable (y) from each other. X contains features that
are related to psychological, behavioral, diagnostic, and trauma-related factors and y is the binary indicator for whether the
student align with optimal sleep duration or not."""
X = df_filtered[['N3Q1', 'N3Q42B', 'N3Q48', 'N3Q20D', 'N3Q20F', 'N3Q20G', 'N3Q65A2', 'N3Q65A3', 
               'N3Q65A7', 'N3Q65A15', 'N3Q65A28', 'N3Q65A35', 'RULS3']]  
y = df_filtered['sleep_binary']

# split the data into training and test sets, stratify=y balances train/test proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=67)

"""The following line of code creates a pipeline for the model. SimpleImputer handles missing values by replacing them with
the mean of the column values. StandardScaler standardizes features to make scaling consistent. LogisticRegression is the model
being used here to predict the probability of being in the optimal sleep category, using a max of 1000 iterations."""
model = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000))  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)                  # prediction of class 0 or class 1 as defined above
y_proba = model.predict_proba(X_test)[:, 1]     # probability of being in optimal sleep range for each test case (1)

# evaluation metrics
print("Accuracy: ", accuracy_score(y_test, y_pred))      # proportion of total correct predictions
print("ROC-AUC: ", roc_auc_score(y_test, y_proba))       # how well the model separates the two classes
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# ROC-AUC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve for Logistic Regression Sleep Model")
plt.show()

# extract the logistic regression step from the pipeline
log_reg = model.named_steps['logisticregression']

# see how much each feature contributes to the prediction, coefficients > 0 increase the odds of optimal sleep and vice versa
coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0],
})

# convert data to odds ratios for easier interpretation, where an odds ratio > 1 means that the predictor increases probability of optimal sleep
coefs['Odds_Ratio'] = np.exp(coefs['Coefficient'])
coefs = coefs.sort_values(by='Odds_Ratio', ascending=False)
print(coefs)

# compute a confusion matrix for easier interpretation
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: {cm}")

# visualize the confusion matrix and provide clearer labels
cmdisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Suboptimal Sleep (0)', 'Optimal Sleep (1)'])
cmdisplay.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Logistic Regression Sleep Model')
plt.show()



# decision tree implementation
print("\nDECISION TREE IMPLEMENTATION\n")

"""Below is the decision tree classification pipeline. Like the logistic regression model, missing values will be replaced with
the column mean using SimpleImputer(strategy='mean'). Gini is used to determine the purity of the data. Having no max_depth value
of None allows the tree to grow fully."""
tree = make_pipeline(SimpleImputer(strategy='mean'), DecisionTreeClassifier(
    criterion='gini', max_depth=None, min_samples_split=2, random_state=67))

tree.fit(X_train, y_train)

# calculate predictions and performance metrics for the tree model
y_pred = tree.predict(X_test)
print(classification_report(y_test, y_pred))

# visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(cm, display_labels=['Suboptimal (0)', 'Optimal (1)']).plot(cmap='Purples', values_format='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()



# random forest implementation
print("\nRANDOM FOREST IMPLEMENTATION\n")

"""Below is the random forest pipeline. SimpleImputer(strategy='mean') will again
fill missing values with the column mean. RandomForestClassifier creates a model
utilizing many decision trees."""
rf_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    RandomForestClassifier(
        n_estimators=300,          # number of trees
        max_depth=None,            # allow full tree growth
        min_samples_split=2,       # default split rule
        random_state=67,           # for reproducibility
        class_weight="balanced"    # handles imbalance in sleep_binary
    )
)

# train the model
rf_model.fit(X_train, y_train)

# predictions
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# evaluation metrics
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print(classification_report(y_test, rf_pred))

# confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)
display = ConfusionMatrixDisplay(rf_cm, display_labels=['Suboptimal (0)', 
                                                        'Optimal (1)'])
display.plot(cmap='Greens', values_format='d')
plt.title("Random Forest Confusion Matrix")
plt.show()


# FEATURE IMPORTANCE PLOT

# extract the trained random forest object
rf = rf_model.named_steps['randomforestclassifier']

importances = rf.feature_importances_
feature_names = X.columns

# package into a DataFrame
rf_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(rf_imp['Feature'], rf_imp['Importance'])
plt.xlabel("Feature Importance Score")
plt.title("Random Forest Feature Importance")
plt.show()



"""Shown below is the attempt at integrating LASSO with the logistic regression model, which did not modify results at all 
and is mostly omitted from the report."""

'''
lasso_model = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), 
                            LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=67))

lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Suboptimal (0)', 'Optimal (1)']).plot(cmap='Purples', values_format='d')
plt.show()
'''



'''
AI assistance was used to suggest solutions and resolve errors
'''