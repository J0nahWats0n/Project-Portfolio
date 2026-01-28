'''
Jonah Watson
Fall 2025
Sleep Health in College Students: A Multivariable Predictive Modeling Analysis
This file was used to profile the ACHA-NCHA Spring 2024 survey data used in this project.
'''

import pandas as pd                 # imported for data manipulation and analysis
import seaborn as sns               # imported for data visualization
import matplotlib.pyplot as plt     # imported for data visualization
import numpy as np                  # imported to perform operations on arrays



# load the cleaned dataset
df = pd.read_csv("CLEANED SLEEP NCHA-III S24 - New_Numeric.csv")



# sleep variables
sleep_vars = ["N3Q13_recode", "N3Q14", "N3Q15"]
# the following list was also used with "sleep_vars" before the impact of the variables within were found to be negligible:
# ["N3Q16A_recode", "N3Q16B_recode", "N3Q16C_recode", "N3Q16D_recode", "N3Q16E_recode"]

# demographic variables
demographic_vars = ["N3Q67A", "N3Q72", "N3Q75A1", "N3Q75A2", "N3Q75A3",
"N3Q75A4", "N3Q75A5", "N3Q75A6", "N3Q75A7", "N3Q75A8", "N3Q77B", "N3Q80",
"RBMI_recode"]

# diagnosis variables
diagnosis_vars = ["N3Q65A2", "N3Q65A3", "N3Q65A7", "N3Q65A15", "N3Q65A28", "N3Q65A35", "N3Q65Y"]

# sleep disorder variables
sleep_disorder_vars = ["N3Q65A28", "N3Q65A35"]

# outcome variables
outcome_vars = ["N3Q1", "N3Q3E_recode", "N3Q3I_recode", "N3Q20D", "N3Q20F", "N3Q20G", "N3Q41C_recode", "N3Q42B", 
"N3Q48", "N3Q66P", "RKESSLER6",  "RULS3", "DIENER"]



# calculate the correlations between sleep variables and demographic variables
sleep_demographic_file = "sleep_demographic_correlations.txt"

"""The following block of code underneath the "with" statement will be reused and slightly modified several more times for each
of the different variable classes that were created above."""
with open(sleep_demographic_file, 'w') as f:
    for demographic in demographic_vars:
        for sleep_var in sleep_vars:

            # write the percentages within each demographic
            percentages = pd.crosstab(df[sleep_var], df[demographic], normalize='columns') * 100 
            f.write(f"\n{sleep_var} vs {demographic}\n")
            f.write(percentages.round(2).to_string())       # converting to string because the data is being written to a file
            f.write("\n\n")

            # write the mean value of demographic variables for each category of sleep variable
            means = df.groupby(demographic)[sleep_var].mean()
            f.write(f"\nMean {sleep_var} per {demographic}:")
            f.write(percentages.round(2).to_string())
            f.write("\n\n" + "="*40 + "\n")            # divider for formatting
    print(f"Correlation analysis saved to {sleep_demographic_file}\n\n")



# calculate and visualize the correlations between different GPAs and time taken to fall asleep
means = df.groupby('N3Q80')['N3Q13_recode'].mean()      
means.plot(kind='bar', title='Mean of Time to Fall Asleep by GPA', color='mediumslateblue')

plt.ylim(1, 3)
y_bins = ("0-15 minutes", "16-30 minutes", "31+ minutes")
plt.yticks([1, 2, 3], y_bins)
plt.ylabel("Mean Time to Fall Asleep")

x_bins = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
plt.xticks(ticks=range(len(x_bins)), labels=x_bins, rotation=0)
plt.xlabel("GPA")

plt.tight_layout()
plt.savefig("mean_time_to_fall_asleep_by_GPA.png")



# calculate and visualize the correlations between different GPAs and average weeknight sleep
means = df.groupby('N3Q80')['N3Q14'].mean()
print(f"Means: {means}")
means.plot(kind='bar', title='Average Weeknight Sleep by GPA', color='mediumslateblue')

y_bins = ("<4 hours", "4 hours", "5 hours", "6 hours", "7 hours", "8 hours", "9 hours", ">9 hours",)
plt.yticks([1, 2, 3, 4, 5, 6, 7, 8], y_bins)
plt.ylabel("Average Weeknight Sleep")

x_bins = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
plt.xticks(ticks=range(len(x_bins)), labels=x_bins, rotation=0)
plt.xlabel("GPA")

plt.tight_layout()
plt.savefig("mean_weeknight_sleep_by_GPA.png")




# calculate the correlations between sleep variables and diagnosis variables
sleep_diagnosis_file = "sleep_diagnosis_correlations.txt"

with open(sleep_diagnosis_file, 'w') as f:
    for diagnosis in diagnosis_vars:
        for sleep_var in sleep_vars:
            percentages = pd.crosstab(df[sleep_var], df[diagnosis], normalize='columns') * 100
            f.write(f"\n{sleep_var} vs {diagnosis}\n")
            f.write(percentages.round(2).to_string())
            f.write("\n\n")

            means = df.groupby(diagnosis)[sleep_var].mean()
            f.write(f"\nMean {sleep_var} per {diagnosis}:")
            f.write(means.round(2).to_string())
            f.write("\n\n" + "="*40 + "\n")
    print(f"Correlation analysis saved to {sleep_diagnosis_file}\n\n")



# calculate the correlations between sleeping disorders (sleep apnea and insomnia) and outcome variables
sleep_disorder_file = "sleep_disorder_correlations.txt"

with open(sleep_disorder_file, 'w') as f:
    for outcome in outcome_vars:
        for sleep_var in sleep_disorder_vars:
            percentages = pd.crosstab(df[sleep_var], df[outcome], normalize='columns') * 100
            f.write(f"\n{sleep_var} vs {outcome}\n")
            f.write(percentages.round(2).to_string())
            f.write("\n\n")

            means = df.groupby(outcome)[sleep_var].mean()
            f.write(f"\nMean {sleep_var} per {outcome}:")
            f.write(means.round(2).to_string())
            f.write("\n\n" + "="*40 + "\n")
    print(f"Correlation analysis saved to {sleep_disorder_file}")



# calculate the correlations between sleeping disorders (sleep apnea and insomnia) and the included mental disorders
dbd_file = "disorder_by_disorder_correlations.txt"

with open(dbd_file, 'w') as f:
    for diagnosis in diagnosis_vars:
        for sleep_var in sleep_disorder_vars:
            percentages = pd.crosstab(df[sleep_var], df[diagnosis], normalize='columns') * 100
            f.write(f"\n{sleep_var} vs {diagnosis}\n")
            f.write(percentages.round(2).to_string())
            f.write("\n\n")

            means = df.groupby(diagnosis)[sleep_var].mean()
            f.write(f"\nMean {sleep_var} per {diagnosis}:")
            f.write(means.round(2).to_string())
            f.write("\n\n" + "="*40 + "\n")
    print(f"Correlation analysis saved to {dbd_file}")



# an empty dictionary to count students that reported having certain diagnoses (ADD/ADHD, anxiety, depression, sleep disorders, etc.)
results = {}

for diagnosis in diagnosis_vars:
    if diagnosis == "N3Q65Y":   # this variable has its own analysis and visualization (wasn't included in my "classes" above)
        continue
    counts = df[diagnosis].dropna().value_counts(normalize=True) * 100
    counts = counts.reindex([1,2])      # responses were reindexed for the sake of analysis
    results[diagnosis] = counts

    # create a new dataframe from the results dictionary and fill missing columns with 0
    diagnosis_counts = pd.DataFrame(results).T.fillna(0)
    diagnosis_counts.columns = ["% who said No (1)", "% who said Yes (2)"]

    """prepare a text file to store the information from survey question N3Q65Y (how many students out of the total survey size have
    been diagnosed with the disorders from the diagnosis_vars list created above)"""
    diagnosis_counts_file = "diagnosis_impacts.txt"

    with open(diagnosis_counts_file, 'w') as f:
        f.write("Percentage of students who said No (1) or Yes (2) for each category:\n\n")
        f.write(diagnosis_counts.round(2).to_string())
        f.write("\n\n" + "="*40 + "\n")

print(f"Diagnosis counts saved to {diagnosis_counts_file}\n\n")

# visualize the counts of each diagnosis
plot_df = diagnosis_counts.reset_index().rename(columns={'index': 'Diagnosis'})

x_labels = {
    "N3Q65A2": "ADD/ADHD",
    "N3Q65A3": "Alcohol/Drug Use Disorder",
    "N3Q65A7": "Anxiety",
    "N3Q65A15": "Depression",
    "N3Q65A28": "Insomnia",
    "N3Q65A35": "Sleep Apnea"
}
plot_df["Diagnosis"] = plot_df["Diagnosis"].replace(x_labels)

plt.figure(figsize=(6,6))
sns.barplot(data=plot_df, x='Diagnosis', y='% who said Yes (2)', color='#bd735b')

plt.title('Percentage of Students with Each Diagnosis (Yes Responses)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('% Yes')
plt.ylim(0, 100)
plt.xlabel('Diagnosis Variable')
plt.tight_layout()
plt.savefig("students_by_diagnosis.png")



# calculate the count of responses for the question "N3Q66P: Last 12 months, have sleep difficulties affected academic performance?
print("Response counts for N3Q66P: \n")
print("did not experience 1, experienced and no affect 2, experienced and bad class performance 3, experience - delayed degree 4")
df = df[df["N3Q66P"].notna()]
print(df["N3Q66P"].value_counts(normalize=True) * 100)



# calculate the correlations between sleep variables and outcome variables
sleep_outcome_file = "sleep_outcome_correlations.txt"

with open(sleep_outcome_file, 'w') as f:
    for outcome in outcome_vars:
        for sleep_var in sleep_vars:
            # percentages for each outcome
            percentages = pd.crosstab(df[sleep_var], df[outcome], normalize='columns') * 100
            f.write(f"\n{sleep_var} vs {outcome}\n")
            f.write(percentages.round(2).to_string())
            f.write("\n\n")

            # mean value of outcome variables for each category of sleep variable
            means = df.groupby(outcome)[sleep_var].mean()
            f.write(f"\nMean {sleep_var} per {outcome}:")
            f.write(means.round(2).to_string())
            f.write("\n\n" + "="*40 + "\n")
    print(f"Correlation analysis saved to {sleep_outcome_file}")


# ===============
# Visualizations
# ===============


# use a stacked bar chart to visualize the relationship between overall health ratings and time taken to fall asleep
health_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
# the values for every visualization come from the correlation text files created above
fall_asleep_time = {
   "<15min": np.array([21.81,  28.36,  37.21,  46.68,  53.72]),
    "16-30min": np.array([20.04,  24.97,  28.97,  29.22,  26.80]),
    "31+min": np.array([58.15,  46.67,  33.82,  24.10,  19.47]),
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(health_labels))

"""The following four lines of code create a stacked bar chart where each category on the y-axis gets its own color and is
stacked on top previous categories. This format is used for each of the following stacked bar chart visualizations as well"""
colors = plt.cm.summer(np.linspace(0, 1, len(fall_asleep_time)))
for i, (timerange, percent) in enumerate(fall_asleep_time.items()):
    ax.bar(health_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Time Needed to Fall Asleep by Overall Health Rating', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Overall Health Rating', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Time to Fall Asleep', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('time_to_fall_asleep_by_overall_health.png')



# use a heatmap to visualize the relationship between overall health ratings and time taken to fall asleep
df_sleep_health = pd.DataFrame(fall_asleep_time, index=health_labels)
df_sleep_health = df_sleep_health.T       # .T transposes the data so that the durations are rows, and the health ratings are columns

plt.figure(figsize=(9,6))
sns.heatmap(
    df_sleep_health,
    cmap="Greens",        
    annot=True,            # Show values inside cells
    fmt=".1f",             # One decimal place
    cbar_kws={'label': 'Percentage of Students (%)'}
)

plt.title('Time Needed to Fall Asleep by Overall Health Rating', fontsize=14, weight='bold')
plt.xlabel('Overall Health Rating', fontsize=12)
plt.ylabel('Percentage of Students (%)', fontsize=12)
plt.tight_layout()
plt.savefig('heatmap_time_to_fall_asleep_by_overall_health.png')



# use a stacked bar chart to visualize the relationship between overall health ratings and weeknight sleep duration
health_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
weeknight_sleep_durations = {
    "<=4h": np.array([6.54, 2.72, 1.10, 0.59, 1.05]),
    "4h": np.array([8.79, 6.04, 3.45, 1.93, 2.19]),
    "5h": np.array([20.84, 17.68, 13.23, 8.68, 7.74]),
    "6h": np.array([27.32, 30.71, 28.77, 24.73, 22.18]),
    "7h": np.array([17.64, 25.08, 32.27, 36.66, 34.80]),
    "8h": np.array([11.17, 11.74, 16.15, 21.97, 26.04]),
    "9h": np.array([3.81, 3.83, 3.67, 4.37, 4.44]),
    ">9h": np.array([3.88, 2.21, 1.37, 1.06, 1.56]),
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(health_labels))

colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(weeknight_sleep_durations)))
for i, (timerange, percent) in enumerate(weeknight_sleep_durations.items()):
    ax.bar(health_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Average Weeknight Sleep Duration by Overall Health Rating', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Overall Health Rating', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Average Weeknight Sleep Duration', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('average_weeknight_sleep_by_overall_health.png')



# use a heatmap to visualize the relationship between overall health ratings and weeknight sleep duration
df_sleep_health = pd.DataFrame(weeknight_sleep_durations, index=health_labels)
df_sleep_health = df_sleep_health.T  

plt.figure(figsize=(9,6))
sns.heatmap(
    df_sleep_health,
    cmap="Greens",         
    annot=True,            
    fmt=".1f",             
    cbar_kws={'label': 'Percentage of Students (%)'}
)

plt.title('Average Weeknight Sleep Duration by Overall Health Rating', fontsize=14, weight='bold')
plt.xlabel('Overall Health Rating', fontsize=12)
plt.ylabel('Average Weeknight Sleep Duration',fontsize=12)
plt.tight_layout()
plt.savefig('heatmap_weeknight_sleep_by_overall_health.png')



# use a stacked bar chart to visualize the relationship between time spent partying and weekend night sleep duration
partying_per_week_labels = ['0-5 hrs', '6-15 hrs', '16-25 hrs', '26+ hrs']
weekend_sleep_durations = {
    "<=4h": np.array([0.82, 1.47, 3.34, 6.29]),
    "4h": np.array([1.74, 3.60, 7.69, 8.61]),
    "5h": np.array([5.10, 8.91, 13.04, 14.57]),
    "6h": np.array([12.38, 15.82, 18.28, 14.90]),
    "7h": np.array([21.42, 20.29, 19.73, 17.22]),
    "8h": np.array([30.52, 25.93, 18.17, 17.88]),
    "9h": np.array([19.32, 16.24, 10.81, 7.62]),
    ">9h": np.array([8.70, 7.74, 8.92, 12.91])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(partying_per_week_labels))

colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(weekend_sleep_durations)))
for i, (timerange, percent) in enumerate(weekend_sleep_durations.items()):
    ax.bar(partying_per_week_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Average Weekend Sleep Duration by Time Spent Partying Per Week', fontsize=13, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Time Spent Partying Per Week', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Average Weekend Night Sleep Duration', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('average_weekend_sleep_by_partying_per_week.png')



# use a stacked bar chart to visualize the relationship between rape trauma and weekend night sleep duration
rape_trauma_labels = ['Not a victim of rape (last 12 months)', 'Victims of rape (last 12 months)']
weekend_sleep_durations = {
    "<=4h": np.array([0.88, 3.43]),
    "4h": np.array([1.90, 6.46]),
    "5h": np.array([5.39, 10.51]),
    "6h": np.array([12.65, 16.62]),
    "7h": np.array([21.38, 17.38]),
    "8h": np.array([30.15, 21.63]),
    "9h": np.array([19.02, 15.11]),
    ">9h": np.array([8.64, 8.86])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(rape_trauma_labels))

colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(weekend_sleep_durations)))
for i, (timerange, percent) in enumerate(weekend_sleep_durations.items()):
    ax.bar(rape_trauma_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Average Weekend Sleep Duration by Presence of Rape Trauma', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Presence of Rape Trauma', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Average Weekend Night Sleep Duration', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('average_weekend_sleep_by_rape_trauma.png')



# use a stacked bar chart to visualize the relationship between resilience level and weeknight sleep duration
resilience_labels = ['not resilient', 'rarely resilient', 'sometimes resilient', 'often resilient', 'very resilient']
weeknight_sleep_durations = {
   "<=4h": np.array([6.03,  2.64,  1.51,  0.87,  1.04]),
    "4h":   np.array([7.07,  5.35,  4.09,  2.77,  2.73]),
    "5h":   np.array([15.49, 14.45, 13.81, 10.87, 10.72]),
    "6h":   np.array([25.61, 29.94, 28.62, 26.41, 25.29]),
    "7h":   np.array([22.26, 26.08, 30.80, 34.62, 33.74]),
    "8h":   np.array([15.41, 14.37, 15.51, 19.24, 21.12]),
    "9h":   np.array([4.54,  4.55,  3.85,  4.01,  4.12]),
    ">9h":  np.array([3.57,  2.64,  1.79,  1.20,  1.24])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(resilience_labels))

colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(weeknight_sleep_durations)))
for i, (timerange, percent) in enumerate(weeknight_sleep_durations.items()):
    ax.bar(resilience_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Average Weeknight Sleep Duration by Level of Resilience', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Level of Resilience', fontsize=12)
plt.xticks(rotation=45)
ax.set_ylim(0, 100)
ax.legend(title='Average Weeknight Sleep Duration', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('average_weeknight_sleep_by_resilience.png')



# use a heatmap to visualize the relationship between resilience level and weeknight sleep duration
df_sleep_health = pd.DataFrame(weeknight_sleep_durations, index=resilience_labels)
df_sleep_health = df_sleep_health.T         

plt.figure(figsize=(9,6))
sns.heatmap(
    df_sleep_health,
    cmap="Greens",         
    annot=True,            
    fmt=".1f",             
    cbar_kws={'label': 'Percentage of Students (%)'}
)

plt.title('Average Weeknight Sleep Duration by Level of Resilience', fontsize=14, weight='bold')
plt.xlabel('Overall Health Rating', fontsize=12)
plt.ylabel('Average Weeknight Sleep Duration',fontsize=12)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('heatmap_weeknight_sleep_by_resilience.png')



# use a stacked bar chart to visualize the relationship between stress and weeknight sleep duration
stress_labels = ['no stress', 'low stress', 'moderate stress', 'high stress']
weeknight_sleep_durations = {
    "<=4h": np.array([3.39,   0.58,  0.69,   2.50]),
    "4h":   np.array([5.41,   1.56,   2.40,   5.92]),
    "5h":   np.array([9.62,   6.29,  10.62, 18.09]),
    "6h":   np.array([21.00,  21.36,  27.17,  30.39]),
    "7h":   np.array([25.21,  37.75,  34.84,  26.21]),
    "8h":   np.array([25.70,  25.95,  18.87,  12.46]),
    "9h":   np.array([6.12,   5.17,   4.08,  2.92]),
    ">9h":  np.array([3.55,   1.35,   1.33,   1.51])
}

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
bottom = np.zeros(len(stress_labels))

colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(weeknight_sleep_durations)))
for i, (timerange, percent) in enumerate(weeknight_sleep_durations.items()):
    ax.bar(stress_labels, percent, width, label=timerange, bottom=bottom, color=colors[i])
    bottom += percent

ax.set_title('Average Weeknight Sleep Duration by Level of Stress', fontsize=14, weight='bold')
ax.set_ylabel('Percentage of Students (%)', fontsize=12)
ax.set_xlabel('Level of Stress', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Average Weeknight Sleep Duration', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('average_weeknight_sleep_by_stress.png')



# use a least-squares polynomial regression to visualize the relationship between resilience level and weeknight sleep duration
diener_scores = np.array([
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56
])
weeknight_sleep_means = np.array([
    2.20, 2.26, 2.00, 1.79, 1.78, 2.05, 2.03, 1.96, 2.05, 1.88, 2.04, 2.09,
    2.12, 2.21, 2.14, 2.05, 2.15, 2.11, 2.19, 2.15, 2.16, 2.14, 2.19, 2.17,
    2.24, 2.25, 2.29, 2.30, 2.29, 2.31, 2.35, 2.38, 2.39, 2.38, 2.41, 2.46,
    2.44, 2.46, 2.50, 2.53, 2.56, 2.56, 2.56, 2.58, 2.57, 2.58, 2.57, 2.64, 2.61
])

width = 0.6

fig, ax = plt.subplots(figsize=(10,6))
z = np.polyfit(diener_scores, weeknight_sleep_means, 1)     # perform a least-squares polynomial regression fit with the given data points
p = np.poly1d(z)                                            # create an polynomial function object to be used below
ax.plot(diener_scores, weeknight_sleep_means, color='royalblue', marker='o')        # plot the original data points as a scatter line
ax.plot(diener_scores, p(diener_scores), color='green', linestyle='--', label='Trend line')     # plot the trend line produced by the least-squares polynomial regression
ax.set_title('Mean Weeknight Sleep vs. DIENER Score', fontsize=14, weight='bold')
ax.set_xlabel('DIENER Score (Life Satisfaction: Higher number = higher quality of well-being)', fontsize=12)
ax.set_ylabel('Mean Weeknight Sleep Duration', fontsize=12)
ax.set_ylim(1.5, 2.9)
ax.grid(alpha=0.3)

ax.legend()
plt.tight_layout()
plt.savefig('average_weeknight_sleep_vs_diener_scores.png')



# use a barplot to visualize the relationship between students with each disorder that either also reported not having sleep apnea or did also report having sleep apnea
data = {
    'Disorder': ['ADHD', 'Substance Abuse', 'Anxiety', 'Depression', 'Insomnia', 'Academic Impact'],
    'Without_Disorder': [1.86, 2.2, 1.28, 1.26, 1.78, 2.43],    # percent of students with each disorder that did not also report sleep apnea
    'With_Disorder': [5.44, 12.1, 4.38, 5.34, 9.93, 5.66]       # percent of students with each disorder that did also report sleep apnea
}

df = pd.DataFrame(data)

melted_df = df.melt(id_vars='Disorder', value_vars=['Without_Disorder', 'With_Disorder'], var_name='Group', value_name='Percent')
# melt gives variables two columns each

colors = {'Without_Disorder': 'silver', 'With_Disorder': 'indianred'}

plt.figure(figsize=(9,5))
sns.barplot(x='Disorder', y='Percent', hue='Group', data=melted_df, palette=colors)

plt.title('Prevalence of Sleep Apnea by Mental/Behavioral Disorder', weight="bold", fontsize="medium")
plt.ylabel('Percent of Students Reporting Sleep Apnea (%)')
plt.xlabel('Diagnosis')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Group')
plt.ylim(0, 35)

plt.tight_layout()
plt.savefig('sleep_apnea_and_other_disorders.png')



# use a barplot to visualize the relationship between students with each disorder that either also reported not having insomnia or did also report having insomnia
data = {
    'Disorder': ['ADHD', 'Substance Abuse', 'Anxiety', 'Depression', 'Sleep Apnea', 'Academic Impact'],
    'Without_Disorder': [5.14, 6.75, 1.87, 2.37, 6.6, 5.27],    # percent of students WITHOUT the disorder who report insomnia
    'With_Disorder': [19.72, 33.33, 16.95, 20.34, 30.1, 21.64]  # percent of students WITH the disorder who report insomnia
}

df = pd.DataFrame(data)

melted_df = df.melt(id_vars='Disorder', value_vars=['Without_Disorder', 'With_Disorder'], var_name='Group', value_name='Percent')
# melt gives variables two columns each

colors = {'Without_Disorder': 'silver', 'With_Disorder': 'dodgerblue'}

plt.figure(figsize=(9,5))
sns.barplot(x='Disorder', y='Percent', hue='Group', data=melted_df, palette=colors)

plt.title('Prevalence of Insomnia by Mental/Behavioral Disorder', weight="bold", fontsize="medium")
plt.ylabel('Percent of Students Reporting Insomnia (%)')
plt.xlabel('Diagnosis')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Group')
plt.ylim(0, 35)

plt.tight_layout()
plt.savefig('insomnia_and_other_disorders.png')



'''
AI assistance was used to suggest solutions and resolve errors
'''