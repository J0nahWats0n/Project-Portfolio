'''
Jonah Watson
Spring 2026
Mental and Behavioral Consequences of Alcohol Consumption Among College Students
This file was used to clean the ACHA-NCHA Spring 2024 survey data used in this project
'''

import pandas as pd     # imported for data manipulation and analysis
import json             # imported to access the JSON file that stores column label mappings



# open the numeric CSV file
df = pd.read_csv("NCHA-III S24 - New_Numeric.csv")

# open the original CSV file to access the text responses for N3Q28 and fix a mapping issue
df28 = pd.read_csv("NCHA-III S24 - Labeled.csv", usecols=["N3Q28"], keep_default_na=False)

# load the JSON label mappings
with open("NCHA-IIIb labels S24 - Cleaned.json", "r") as f:
    labels = json.load(f)

mapping_28 = labels["N3Q28"]
# reverse mapping: text to numeric, stripping whitespace and converting to lowercase for consistency
reverse_map = {v.strip().lower(): int(float(k)) for k, v in mapping_28.items()}

# this function handles missing inputs and maps text responses to numeric codes
def map_n3q28(val):
    val_str = str(val).strip().lower()      # normalize the input value 
    if val_str == "":                       # check for empty strings
        return pd.NA               
    return reverse_map.get(val_str, pd.NA) 
    # look for standardized value in the reverse mapping dictionary and return pd.NA if not found

# apply the mapping function to the N3Q28 column
df["N3Q28"] = df28["N3Q28"].apply(map_n3q28)

# keep specific columns (removing unnecessary variables from the dataset)
keep_variables = ["N3Q1", "N3Q22B2", "N3Q22K2","N3Q22L2",
    "N3Q22M2","N3Q22N2","N3Q22O2","N3Q25B1","N3Q25B2","N3Q28", "N3Q29A","N3Q29B",
    "N3Q29C","N3Q29D","N3Q29E","N3Q29F","N3Q29G","N3Q29H","N3Q29I","N3Q29J","N3Q29K", "N3Q29L",
    "N3Q30A", "N3Q30B", "N3Q46", "N3Q48", "N3Q65A2","N3Q65A3","N3Q65A7", "N3Q65A15", "N3Q65A19",
    "N3Q65A28","N3Q65A31", "N3Q65A33","N3Q65A35","N3Q65Y","N3Q67A", "N3Q69", "N3Q72", 'N3Q75A1',
    'N3Q75A2', 'N3Q75A3', 'N3Q75A4', 'N3Q75A5', 'N3Q75A6', 'N3Q75A7', 'N3Q75A8', "N3Q77A",
    "N3Q77B", "N3Q80", "RKESSLER6","RULS3","RSBQR","DIENER", "CDRISC2", "ALCOHOLRISK"]

# save the variables I want to keep for analysis
df_cleaned = df[keep_variables]

# save the cleaned numeric CSV
df_cleaned.to_csv("CLEANED ALCOHOL NCHA-III S24 - New_Numeric.csv", index=False)

# check that 'None' responses are now coded as 1 - should be 24530
print(df_cleaned["N3Q28"].value_counts(dropna=False))
print('Cleaned data saved')

        