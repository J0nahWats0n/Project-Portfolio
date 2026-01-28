'''
Jonah Watson
Fall 2025
Sleep Health in College Students: A Multivariable Predictive Modeling Analysis
This file was used to clean the ACHA-NCHA Spring 2024 survey data used in this 
project
'''

import pandas as pd             # imported for data manipulation and analysis



# open the original CSV file
df = pd.read_csv("SLEEP NCHA-III S24 - New_Numeric.csv")

# keep specific columns (removing unnecessary variables)
keep_variables = [
    'N3Q1', 'N3Q3E', 'N3Q3I', 'N3Q13', 'N3Q14', 'N3Q15', 'N3Q16A', 'N3Q16B', 
    'N3Q16C', 'N3Q16D', 'N3Q16E', 'N3Q20D', 'N3Q20F', 'N3Q20G', 'N3Q41C', 
    'N3Q42B', 'N3Q48', 'N3Q65A2', 'N3Q65A3', 'N3Q65A7', 'N3Q65A15', 'N3Q65A28',
    'N3Q65A35', 'N3Q65Y','N3Q66P', 'N3Q67A', 'N3Q69', 'N3Q72', 'N3Q75A1', 
    'N3Q75A2', 'N3Q75A3', 'N3Q75A4', 'N3Q75A5', 'N3Q75A6', 'N3Q75A7', 
    'N3Q75A8', 'N3Q77B', 'N3Q80', 'RBMI', 'RKESSLER6', 'RULS3', 'DIENER']

# save the variables I want to keep for analysis
df = df[keep_variables]



"""This next section shows the process of recoding responses for certain
variables to reduce the number of categories and allow for a more easily 
digestable analysis. All data had already been converted to numeric at this
point"""

def recode_NQ3E_I(x):
    """The variables N3Q3E and N3Q3I represent a student's time spent doing
    physical activity and time spent partying per week, respectively. This 
    function recodes the categories into larger bins"""
    if x in [1, 2]:         # 0-5 hours
        return 1
    elif x in [3, 4]:       # 6-15 hours
        return 2
    elif x in [5, 6]:       # 16-25 hours
        return 3
    elif x in [7, 8]:       # 26+ hours
        return 4
    return pd.NA            # if the response does not belong

def recode_NQ13(x):
    """The variable N3Q13 represents how long it takes a student to fall 
    asleep. This function recodes the categories into larger bins"""
    if x in [1, 2]:         # <15 minutes
        return 1
    elif x == 3: 
        return 2            # 16-30 minutes
    elif x in [4, 5]:       # >30 minutes
        return 3
    return pd.NA

def recode_NQ16A_E(x):
    """The variables N3Q16A, N3Q16B, N3Q16C, N3Q16D, and N3Q16E represent
    questions that relate to a student's sleeping habits, based on the last 
    7 days. This function recodes the categories into larger bins. For more
    specific information on the questions that these variables represent, 
    please see the Data Collection and Profiling subsection under Methodology 
    in the research report."""
    if x in [1, 2]:         # 0-1 days
        return 1
    elif x in [3, 4]:       # 2-3 days
        return 2
    elif x in [5, 6, 7, 8]: # 4+ days
        return 3
    return pd.NA

def recode_NQ41C(x):
    """The variable N3Q41C represents a student's response to if they feel
    that they are engaged and interested in their daily activites. This 
    function recodes the categories into larger bins"""
    if x in [1, 2, 3]:     # disagree
        return 1
    elif x == 4:           # neither
        return 2
    elif x in [5, 6, 7]:   # agree
        return 3
    return pd.NA

def recode_RBMI(x):
    """The variable RBMI represents a student's body mass index. This 
    function recodes the categories into larger bins"""
    if x == 1:             # underweight
        return 1
    elif x == 2:           # healthy weight
        return 2
    elif x == 3:           # overweight
        return 3
    elif x in [4, 5, 6]:   # obese
        return 4
    return pd.NA



# apply recodes to dataframe
df["N3Q3E_recode"] = df["N3Q3E"].apply(recode_NQ3E_I)
df["N3Q3I_recode"] = df["N3Q3I"].apply(recode_NQ3E_I)
df["N3Q13_recode"] = df["N3Q13"].apply(recode_NQ13)

# for loop for recoding these 5 variables since they are in an iterable format
for var in ["N3Q16A", "N3Q16B", "N3Q16C", "N3Q16D", "N3Q16E"]:
    df[f"{var}_recode"] = df[var].apply(recode_NQ16A_E)

df["N3Q41C_recode"] = df["N3Q41C"].apply(recode_NQ41C)
df["RBMI_recode"] = df["RBMI"].apply(recode_RBMI)



# save the cleaned data to a new CSV file
df.to_csv("CLEANED SLEEP NCHA-III S24 - New_Numeric.csv", index=False)

print('Cleaned data saved')

'''
AI assistance was used to suggest solutions and resolve errors
'''