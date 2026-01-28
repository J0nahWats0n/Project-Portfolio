'''
Jonah Watson
Fall 2025
Sleep Health in College Students: A Multivariable Predictive Modeling Analysis
This file was used to create and test the linear regression model used in this research project.
'''

import pandas as pd                                     # imported for data manipulation and analysis
from sklearn.model_selection import train_test_split    # imported to efficiently split the data into training and testing groups
from sklearn.linear_model import LinearRegression       # imported to use the linear regression model
from sklearn.metrics import mean_absolute_error         # imported to calculate the mean absolute error score



# load the cleaned dataset
df = pd.read_csv("CLEANED SLEEP NCHA-III S24 - New_Numeric.csv")



# predictor groups for the linear regression model
psych = ['N3Q42B', 'N3Q48', 'DIENER', 'RKESSLER6', 'RULS3']      
# resilience, stress, well-being/life satisfaction, serious psychological distress, and loneliness

behavior = ['N3Q1', 'N3Q3E', 'N3Q3I', 'N3Q13_recode']
# overall self-rated health, time spent doing physical activity, time spent partying, and time taken to fall asleep

diagnoses = ['N3Q65A2', 'N3Q65A3', 'N3Q65A7', 'N3Q65A15','N3Q65A28', 'N3Q65A35']
# ADHD, substance abuse, anxiety, depression, insomnia, and sleep apnea

trauma = ['N3Q20D', 'N3Q20F', 'N3Q20G']
# sexual assault, rape, and stalking victims



# write predictor groups into a dictionary to properly iterate
groups = {
    "Psychological" : psych,
    "Behavioral" : behavior,
    # "Diagnoses" : diagnoses,
    "Trauma" : trauma
}

# define the target of the prediction (y)
y = df['N3Q14']     # weeknight sleep duration

# cumulative groups
all_predictors = []
results = []

for name, group in groups.items():
    all_predictors.extend(group)        # with each loop iteration, attach the next group that was defined above
    X = df[all_predictors]

    # drop missing predictor and/or target values
    print("Original data size:", len(X))
    df_model = pd.concat([X, y], axis=1).dropna()
    X = df_model[X.columns]
    y_clean = df_model[y.name]
    print("After dropping NaNs:", len(X))

    # split the data into test and train groups - randomly divide the data into 80% used for training and 20% used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size = 0.2, random_state=67)

    # train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)     # model.fit finds the best-fitting coefficients to minimize error in the training data

    # save coefficients for each model (print slope for current predictor group)
    coefficient_table = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })

    print(f"\n--- Coefficients for {group} ---")
    print(coefficient_table)

    # evaluate
    y_prediction = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    n = X_test.shape[0]   # number of samples
    p = X_test.shape[1]   # number of predictors
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    # r-squared measures variance, where 1 is perfect, 0 is as if you guessed randomly, and -1 is worse than guessing the mean
    # "adjusting" r-squared according to this formula allows it to more effectively display if certain variables are contributing to the model or not

    MAE = mean_absolute_error(y_test, y_prediction)
    # MAE measures how far off predictions are, on average, from what the true values are (lower = better)

    results.append({
        "Group Added": name,
        "Number of Predictors": len(all_predictors),
        "Adjusted R²": adjusted_r2,
        "Mean Absolute Error": MAE,
        "Coefficients": coefficient_table
    })

print("\n================")
print("Summary of New Results by Group")
print("================\n")

"""The following for loop shows how much each predictor contributes to the prediction. For example if stress is -0.21 and 
resilience is +0.09 it means that higher stress decreases predicted sleep by -0.21 of a scale point, which, in this case,
where 1 hour more of sleep is 1 scale point means that -0.21 is about 12 minutes less of sleep."""
for result in results:
    print(f"\nResults after {result['Group Added']}:\n")
    print(f"Adjusted R²: {result['Adjusted R²']}, Mean Absolute Error = {result['Mean Absolute Error']}")
    print(result['Coefficients'])
    print("\n")



'''
AI assistance was used to suggest solutions and resolve errors
'''