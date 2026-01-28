'''
Jonah Watson
Spring 2026
Sleep Health in College Students: A Multivariable Predictive Modeling Analysis
This file was used to profile the ACHA-NCHA Spring 2024 survey data used in this project.
'''

import pandas as pd     # imported for data manipulation and analysis
import json             # imported to access the JSON file that stores column label mappings



# load the cleaned dataset
df = pd.read_csv("CLEANED ALCOHOL NCHA-III S24 - New_Numeric.csv")

# load the JSON label mappings
with open("NCHA-IIIb_labels_S24_copycopy.json", 'r') as jsonfile:
    labels = json.load(jsonfile)

# prepare a text file to store profiling data and response rates of each choice for each question
response_rates = "alcohol_study_response_rates.txt"

with open(response_rates, 'w') as f:
    f.write("Number of rows and columns: \n\n")
    f.write(str(df.shape) + "\n\n")

    f.write("General information about the data: \n\n")
    df.info(buf=f)          # buf=f is used redirect df.info() right into the text file
    f.write("\n\n")

    f.write("Quick summary statistics: \n\n")
    f.write(str(df.describe().T) + "\n\n") # .T to flip rows and columns (I find it easier to read)

    f.write("Missing values per column: \n\n")
    f.write(str(df.isnull().sum()) + "\n\n")

    f.write("Count of duplicate rows: \n\n")
    f.write(str(df.duplicated().sum()) + "\n\n")
    df = df.drop_duplicates()
    f.write("Duplicate rows have now been dropped\n\n")
    f.write("Count of duplicate rows is now: \n\n")
    f.write(str(df.duplicated().sum()) + "\n\n")

    f.write("Value counts per column: \n\n")
    
    
    
    for column in df.columns:      
        """This for loop counts the number of responses per column in the dataset by looping through
        every column in the dataset,  counting the frequency of each unique value (including missing
        values), and matching numeric codes to descriptive labels in the JSON file, where the results
        are then written to a text file."""

        f.write(f"\n\n--- {column} ---\n\n") 
        # write the columns name as a section header in the text file

        counts = df[column].value_counts(dropna=False).sort_index()
        # count how many times unique values appear in each column
        # dropna=False included to also count missing values

        variable_name = column.split(":")[0].strip()
        # this is the appropriate way to match variable names based on the way I renamed my columns
        # example:  "N3Q1": "N3Q1: Overall health rating" becomes "N3Q1"

        if variable_name in labels:
        # my JSON file (labels) contains all of the data prior to cleaning
        # so some labels won't be used (that's why I use "if")

            for code, count in counts.items():
            # for the JSON labels associated with the variables that I am using, count each response rate
                
                if pd.isna(code):
                    label = "Missing"
                else:
                    label = labels[variable_name].get(str(int(code)), code)
                # if the data is blank, assign it the name "Missing" in the text file
                # if the data exists, look at the JSON for what the numeric correlation is
                # example. in some questions "1" = "Excellent", "2" = "Very Good", etc.

                f.write(f"{label} ({code}): {count}\n\n")
                # write the results into the text file
        else:
            f.write(str(counts) + "\n\n")
            # if for some reason, the columns aren't able to be matched with the JSON file, still print them

print("Data successfully written to text file")

'''
AI assistance was used to suggest solutions and resolve errors
'''
