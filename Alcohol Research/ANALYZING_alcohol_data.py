'''
Jonah Watson
Spring 2026
Mental and Behavioral Consequences of Alcohol Consumption Among College Students
This file was used to profile the ACHA-NCHA Spring 2024 survey data used in this project.
'''

import pandas as pd                                 # imported for data manipulation and analysis
import matplotlib.pyplot as plt                     # imported for data visualization
import numpy as np                                  # imported to perform operations on arrays
from sklearn.preprocessing import StandardScaler    # imported to standardize the data
from sklearn.decomposition import PCA               # imported to perform Principal Component Analysis (PCA)



# load the cleaned dataset
df = pd.read_csv("CLEANED ALCOHOL NCHA-III S24 - New_Numeric.csv")

# recode this question so that higher values indicate better overall health
# to be consistent for the format of other questions
df['N3Q1'] = 6 - df['N3Q1']    # 1 becomes 5, 2 becomes 4, etc.


# shortened lists of variables for analysis:

# alcohol variables
# alcohol_vars = ["N3Q22B2", "N3Q28", "ALCOHOLRISK"]
alcohol_vars = ["N3Q22L2", "N3Q22O2", "N3Q29B", "N3Q30A"]

# demographic variables
demographic_vars = ["N3Q67A", "N3Q72", "N3Q77A", "N3Q77B", "N3Q80"]

# health variables
health_vars = ["N3Q1", "N3Q48", "N3Q65A2", "N3Q65A7", "N3Q65A15", 
"N3Q65A19", "N3Q65A28","N3Q65A31",
"N3Q65A33","N3Q65A35", "RKESSLER6","RULS3","RSBQR","DIENER", "CDRISC2"]



# prepare a text file to store correlations between alcohol and demographic variables
alc_demographic_file = "alc_demographic_correlations.txt"

"""The following block of code underneath the "with" statement will be reused and
slightly modified again for the health variable class that was created above as well."""
with open(alc_demographic_file, 'w') as f:
    for demographic in demographic_vars:
        for alc in alcohol_vars:

            # write the percentages within each demographic
            percentages = pd.crosstab(df[alc], df[demographic], normalize='columns') * 100 
            f.write(f"\n{alc} vs {demographic}\n")
            f.write(percentages.round(2).to_string()) 
            # converting to string because the data is being written to a file
            f.write("\n\n")

print(f"Correlation analysis saved to {alc_demographic_file}\n\n")



# prepare a text file to store the correlations between alcohol and health variables
alc_health_file = "alc_health_correlations.txt"

with open(alc_health_file, 'w') as f:
    for health in health_vars:
        for alc in alcohol_vars:

            # write the percentages within each demographic
            percentages = pd.crosstab(df[alc], df[health], normalize='columns') * 100 
            f.write(f"\n{alc} vs {health}\n")
            f.write(percentages.round(2).to_string())
            # converting to string because the data is being written to a file
            f.write("\n\n")

print(f"Correlation analysis saved to {alc_health_file}\n\n")


# =======================
# Visualizations
# =======================

# use a stacked bar chart to visualize the relationship between risk of alcohol misuse
# and the PRESENCE of specific disorders
disorder_labels = ['ADD\ADHD\n(n = 14,032)', 'anxiety\n(n = 35,840)', 'depression\n(n = 27,210)',
        'gambling disorder\n(n = 162)', 'insomnia\n(n = 7,239)', 'OCD\n(n = 6,133)',
        'PTSD\n(n = 8,292)', 'sleep apnea\n(n = 2,394)']
alc_risks = {
    "Low Risk": np.array([79.66, 81.79, 80.27, 48.91, 80.06, 79.71, 79.39, 81.78]),
    "Moderate Risk":   np.array([17.82, 16.33, 17.44, 29.35, 16.94, 17.45, 17.48, 14.92]),
    "High Risk":   np.array([2.51, 1.87, 2.29, 21.74, 3.00, 2.84, 3.13, 3.31])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(disorder_labels))

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for timerange, percent in alc_risks.items():
    ax.bar(disorder_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Risk of Alcohol Misuse by PRESENCE of Disorders', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Disorders', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_disorderPRESENCE.png')



# use a stacked bar chart to visualize the relationship between risk of alcohol misuse
# and the ABSENCE of specific disorders
disorder_labels = ['no ADD\ADHD\n(n = 87,472)', 'no anxiety\n(n = 65,895)', 
    'no depression\n(n = 74,446)', 'no gambling disorder\n(n = 101,144)',
    'no insomnia\n(n = 93,560)', 'no OCD\n(n = 95,464)',
    'no PTSD\n(n = 93,279)', 'no sleep apnea\n(n = 98,852)']
alc_risks = {
    "Low Risk":        np.array([84.91, 85.58, 85.82, 84.17, 84.47, 84.42, 84.59, 84.20]),
    "Moderate Risk":   np.array([13.88, 13.31, 13.16, 14.46, 14.27, 14.28, 14.18, 14.45]),
    "High Risk":       np.array([1.21, 1.11, 2.29, 1.37, 1.27, 1.30, 1.22, 1.35])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(disorder_labels))

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for timerange, percent in alc_risks.items():
    ax.bar(disorder_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Risk of Alcohol Misuse by ABSENCE of Disorders', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Disorders', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_disorderABSENCE.png')



# use a stacked bar chart to visualize the relationship between risk of alcohol misuse and GPA
gpa_labels = [
    'A+', 'A', 'A-', 'B+', 'B', 'B-',
    'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F'
]

alc_risks = {
    "Low Risk": np.array([
        86.96, 86.16, 83.47, 82.18, 82.23, 80.36,
        80.20, 81.81, 78.51, 69.68, 73.50, 73.68, 72.22
    ]),
    "Moderate Risk": np.array([
        11.80, 12.82, 15.23, 16.32, 15.97, 17.23,
        17.71, 15.92, 19.74, 25.81, 22.22, 18.42, 12.96
    ]),
    "High Risk": np.array([
        1.23, 1.01, 1.31, 1.51, 1.80, 2.41,
        2.09, 2.27, 1.75, 4.52, 4.27, 7.89, 14.81
    ])
}

width = 0.7
fig, ax = plt.subplots(figsize=(9, 6))
bottom = np.zeros(len(gpa_labels))


colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for risk, percent in alc_risks.items():
    ax.bar(gpa_labels, percent, width, label=risk, bottom=bottom,
        color=colors[risk], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

# Formatting
ax.set_title('Risk of Alcohol Misuse by GPA', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('GPA Category', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_GPA.png')



# use a stacked bar chart to visualize the relationship between risk of alcohol misuse
# and Greek Life participation
greek_labels = ['Not members of a fraternity/sorority', 'Members of a fraternity/sorority']
alc_risks = {
    "Low Risk":        np.array([84.94, 73.69]),
    "Moderate Risk":   np.array([13.70, 24.16]),
    "High Risk":       np.array([1.36, 2.15])
}

width = 0.6

fig, ax = plt.subplots(figsize=(8,8))
bottom = np.zeros(len(greek_labels))

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for timerange, percent in alc_risks.items():
    ax.bar(greek_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Risk of Alcohol Misuse by Greek Life Participation', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Greek Life Participation', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_greek_life_particpation.png')



# use a stacked bar chart to visualize the relationship between frequency of alcohol use
# in the last 3 months and Greek Life living
greek_labels = ['Not living in a fraternity/sorority', 'Living in a fraternity/sorority']
alc_rates = {
    "Never":           np.array([8.74, 3.38]),
    "Once or twice":   np.array([31.37, 13.04]),
    "Monthly":         np.array([26.06, 18.51]),
    "Weekly":          np.array([31.43, 60.54]),
    "Daily/almost":    np.array([2.40, 4.54])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(greek_labels))

colors = {
    "Never": "#4CAF50",         # green
    "Once or twice": "#8BC34A", # light green
    "Monthly": "#FFC107",       # yellow/orange
    "Weekly": "#FF9800",        # orange
    "Daily/almost": "#F44336"   # red
}

for timerange, percent in alc_rates.items():
    ax.bar(greek_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Frequency of Alcohol Consumption by Greek Life Living Situation',
             fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Greek Life Living Situation', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Frequency of Alcohol Consumption\n(last 3 months)',
          loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('alcohol_consumption_by_greek_life_living.png')



# use a stacked bar chart to visualize the relationship between risk of alcohol misuse
# and Greek Life living
greek_labels = ['Not living in a fraternity/sorority', 'Living in a fraternity/sorority']
alc_risks = {
    "Low Risk":        np.array([84.27, 67.36]),
    "Moderate Risk":   np.array([14.35, 27.85]),
    "High Risk":       np.array([1.38, 4.79])
}

width = 0.6

fig, ax = plt.subplots(figsize=(9,6))
bottom = np.zeros(len(greek_labels))

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for timerange, percent in alc_risks.items():
    ax.bar(greek_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Risk of Alcohol Misuse by Greek Life Living Situation', 
             fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Greek Life Living Situation', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.05, 1))
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_greek_life_living.png')



# use a stacked area chart to visualize the relationship between DIENER scores
# and risk of alcohol misuse
diener_scores = np.arange(8, 57)

low = np.array([
    73.73,76.67,78.79,67.86,82.76,75.68,70.97,74.00,57.45,74.65,
    65.43,73.58,75.23,67.74,72.00,73.25,71.01,73.36,76.64,75.32,
    77.15,74.20,75.39,74.75,77.97,80.68,80.00,81.28,79.94,81.12,
    79.52,83.19,81.01,81.50,82.59,83.36,83.97,83.15,84.58,84.14,
    86.46,86.11,86.18,87.21,87.02,86.48,86.13,87.03,88.78
])

moderate = np.array([
    18.64,16.67,18.18,21.43,10.34,21.62,22.58,18.00,35.11,19.72,
    29.63,23.58,18.35,27.42,20.67,24.20,22.69,22.71,18.61,21.47,
    18.18,20.88,20.51,22.20,17.07,16.56,16.61,16.72,17.60,16.61,
    18.88,15.18,16.86,17.05,15.83,15.23,14.68,15.82,14.35,14.91,
    12.53,13.18,13.01,12.12,12.46,13.03,13.16,12.61,10.44
])

high = np.array([
    7.63,6.67,3.03,10.71,6.90,2.70,6.45,8.00,7.45,5.63,
    4.94,2.83,6.42,4.84,7.33,2.55,6.30,3.93,4.74,3.21,
    4.67,4.91,4.10,3.05,4.96,2.76,3.39,2.00,2.47,2.26,
    1.60,1.63,2.13,1.45,1.58,1.42,1.35,1.03,1.07,0.94,
    1.01,0.71,0.82,0.66,0.52,0.50,0.71,0.37,0.79
])

# set up trend line
moderate_high = moderate + high

coef = np.polyfit(diener_scores, moderate_high, 1)
trend_line = np.poly1d(coef)(diener_scores)

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

plt.figure(figsize=(11, 6))

plt.stackplot(diener_scores, high, moderate, low,
    labels=['High Risk', 'Moderate Risk', 'Low Risk'], 
    colors=[colors["High Risk"], colors["Moderate Risk"], colors["Low Risk"]],
    alpha=0.85
)

# plot trend line
plt.plot(
    diener_scores,
    trend_line,
    color='black',
    linewidth=2.5,
    linestyle='--',
    label='Trend: Moderate–High Risk'
)

plt.title('Alcohol Misuse Risk Distribution by Well-Being (DIENER)',
          fontsize=14, weight='bold')
plt.xlabel('DIENER Flourishing Score', fontsize=12)
plt.ylabel('Percentage of Students (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_DIENER_stacked_area.png', dpi=300)



# create a stacked bar chart to visualize the relationship between CDRISC2 scores
# and risk of alcohol misuse
cdrisc2_labels = ['0 (not resilient)', '1', '2', '3', '4 (somewhat resilient)', 
                  '5', '6', '7', '8 (very resilient)']
alc_risks = {
    "Low Risk": np.array([68.15, 77.56, 75.82, 77.52, 80.70, 82.65, 84.87, 85.13, 86.35]),
    "Moderate Risk": np.array([22.18, 20.49, 19.48, 19.53, 16.92, 15.73, 13.07, 13.90, 12.68]),
    "High Risk":   np.array([9.68, 1.95, 4.71, 2.95, 2.38, 1.63, 1.17, 0.97, 0.98])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(cdrisc2_labels))

colors = {
    "Low Risk": "#4CAF50",      # green
    "Moderate Risk": "#FFC107", # yellow/orange
    "High Risk": "#F44336"      # red
}

for timerange, percent in alc_risks.items():
    ax.bar(cdrisc2_labels, percent, width, label=timerange, bottom=bottom,
           color=colors[timerange], edgecolor='black', linewidth=0.6
    )
    bottom = bottom + percent

ax.set_title('Risk of Alcohol Misuse by CDRISC2 Score', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('CDRISC2 Score (Higher Scores Indicate Greater Resilience)', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Risk of Alcohol Misuse', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ALCOHOLRISK_by_CDRISC2.png')



# create a stacked bar chart to visualize the relationship between alcohol-related outcomes
# and Greek Life living situation
questions = [
    'Health/social/legal/financial\nproblems from alcohol',
    'Failed to control\nalcohol use',
    'Blackout while\ndrinking',
    'Drove after\ndrinking'
]

not_greek = np.array([10.7, 8.4, 9.9, 12.8])
greek = np.array([23.8, 10.6, 26.7, 11.6])

x = np.arange(len(questions))
width = 0.6

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "Not Greek": "#4CAF50",   # green
    "Greek": "#F44336"        # red
}

ax.bar(x, not_greek, width, label='Not in Greek housing',
    color=colors["Not Greek"], edgecolor='black', linewidth=0.6
)

ax.bar(x, greek, width, bottom=not_greek,
    label='In Greek housing', color=colors["Greek"], edgecolor='black', linewidth=0.6
)

ax.set_title('Alcohol-Related Outcomes by Greek Life Living Situation', 
             fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Alcohol-Related Outcomes', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(questions, rotation=20, ha='right')
ax.set_ylim(0, 40)
ax.legend(title='Living Situation', loc='upper right')

plt.tight_layout()
plt.savefig('Alcohol_outcomes_by_greek_life_living.png')



# create a stacked bar chart to visualize the relationship between DIENER scores
# and presence of alcohol-related problems
diener_bins = [
    'Very low\n(8–17)',
    'Low\n(18–27)',
    'Moderate\n(28–37)',
    'High\n(38–47)',
    'Very high\n(48–56)'
]

# Percentages (example structure — plug in your computed values)
no_problems = np.array([80.3, 79.9, 84.4, 88.2, 91.4])
any_problems = np.array([19.7, 20.1, 15.6, 11.8, 8.6])

width = 0.7
x = np.arange(len(diener_bins))

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "No problems": "#4CAF50",   # green
    "Any problems": "#F44336"   # red
}

ax.bar(x, no_problems, width, label='No alcohol-related problems',
    color=colors["No problems"], edgecolor='black', linewidth=0.6
)

ax.bar(x, any_problems, width,bottom=no_problems, label='Any alcohol-related problems',
    color=colors["Any problems"], edgecolor='black', linewidth=0.6
)

ax.set_title(
    'Any Alcohol-Related Problems by Well-Being (DIENER)', fontsize=14, weight='bold'
)

ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('DIENER Well-Being Score', fontsize=12)
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(diener_bins)

ax.legend(
    title='Alcohol-Related Problems',
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

plt.tight_layout()
plt.savefig('alcohol_outcomes_by_DIENER.png')

'''
AI assistance was used to suggest solutions and resolve errors
'''