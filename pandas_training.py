import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and display its shape
gallstone_dataset = pd.read_csv("dataset-uci.csv", delimiter=';', decimal=',') # decimal option converts ',' to '.' , objects to floats
print(gallstone_dataset.shape)

# Check colnames and data types
print(gallstone_dataset.columns)
print(gallstone_dataset.dtypes)

# Check for missing values and count per column
print(gallstone_dataset.isnull().sum())

# Preview the first and last 5 rows
print(gallstone_dataset.head())
print(gallstone_dataset.tail())

# Check for duplicates and remove them if any
print(gallstone_dataset.duplicated().sum())
#gallstone_dataset = gallstone_dataset.drop_duplicates()

# Rename some columns to simpler names
gallstone_dataset.rename(columns={'Aspartat Aminotransferaz (AST)': 'AST', 
                                  'Alanin Aminotransferaz (ALT)': 'ALT', 
                                  'C-Reactive Protein (CRP)': 'CRP'}, inplace=True)
print(gallstone_dataset.columns)

# Convert 'Gender' from 0/1 to Male/Female
gallstone_dataset['Gender'].replace({0:'Male', 1:'Female'}, inplace=True)
print(gallstone_dataset['Gender'])

# Convert Diabetes status codes to labels
gallstone_dataset['Diabetes Mellitus (DM)'].replace({0: 'No', 1: 'Yes'}, inplace=True)
print(gallstone_dataset['Diabetes Mellitus (DM)'])

# Check if all rows are unique
print(gallstone_dataset.duplicated().any()) # Should return False if no duplicates exist

# Select rows where BMI > 30
BMI_above_30 = gallstone_dataset[gallstone_dataset['Body Mass Index (BMI)'] > 30]
print(BMI_above_30.shape)

# Filter patients aged above 30 with high triglycerides (> 200)
high_trigl_30 = gallstone_dataset[(gallstone_dataset['Age'] > 30) & 
                                  (gallstone_dataset['Triglyceride'] > 200)]
print(high_trigl_30.shape)
print(high_trigl_30.head())

# Select patients with HFA of 3 or 4
HFA_choice = gallstone_dataset[gallstone_dataset['Hepatic Fat Accumulation (HFA)'].isin([3, 4])]
print(HFA_choice.shape)
print(HFA_choice.head())

# Find female patients with diabetes and gallstones
females_withDG = gallstone_dataset[(gallstone_dataset['Gender'] == 'Female') &
                                   (gallstone_dataset['Diabetes Mellitus (DM)'] == 'Yes') &
                                   (gallstone_dataset['Gallstone Status'] == 1)]

print(females_withDG.shape)
print(females_withDG.head())

# Get rows where cholesterol is missing
missing_chol = gallstone_dataset[gallstone_dataset['Total Cholesterol (TC)'].isnull()]
print(missing_chol.shape)

# Find patients with 2+ comorbidities and low Vitamin D (< 10)
patients_with_high_com_and_low_D = gallstone_dataset[(gallstone_dataset['Comorbidity'] >= 2) &
                                                     (gallstone_dataset['Vitamin D'] < 10)]
print(patients_with_high_com_and_low_D.shape)

# Get patients with BMI between 18.5 and 25
BMI_check = gallstone_dataset[gallstone_dataset['Body Mass Index (BMI)'].between(18.5, 25)]
print(BMI_check['Body Mass Index (BMI)'])

# Find male patients with CRP > 10
male_crp = gallstone_dataset[(gallstone_dataset['Gender'] == 'Male') &
                            (gallstone_dataset['CRP'] > 10)]
print(male_crp[['Gender', 'CRP']])

# List patients with creatinine > 1.2 and GFR < 60
creatin_gfr = gallstone_dataset[(gallstone_dataset['Creatinine'] > 1.2) &
                                (gallstone_dataset['Glomerular Filtration Rate (GFR)'] < 60)]

print(creatin_gfr[['Creatinine', 'Glomerular Filtration Rate (GFR)']])

# Subset the dataframe to only columns related to blood markers
blood_markers = gallstone_dataset[['Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
                                   'High Density Lipoprotein (HDL)', 'Triglyceride',
                                   'AST', 'ALT',
                                   'Alkaline Phosphatase (ALP)', 'Creatinine',
                                   'Glomerular Filtration Rate (GFR)', 'CRP',
                                   'Hemoglobin (HGB)', 'Vitamin D']]

print(blood_markers.head())

# Sort dataframe by Age descending
sorted_by_age = gallstone_dataset.sort_values(by='Age', ascending=False)
print(sorted_by_age.head())

# Sort by BMI and then by Weight
sort_bmi_weight = gallstone_dataset.sort_values(by=['Body Mass Index (BMI)', 'Weight'], ascending=[True, False])
print(sort_bmi_weight[['Body Mass Index (BMI)', 'Weight']]) # If two or more rows have the same BMI, it sorts those rows by Weight (descending)

# Reset the index of the dataframe and set 'Age' as the new one
new_index = gallstone_dataset.reset_index(drop=True)
new_index.set_index('Age', inplace = True)
print(new_index.head())

# Sort rows where HFA is not 0 and order by cholesterol
HFA_not_zero = gallstone_dataset[gallstone_dataset['Hepatic Fat Accumulation (HFA)'] != 0]
HFA_not_zero = HFA_not_zero.sort_values(by='Total Cholesterol (TC)', ascending=True)
print(HFA_not_zero[['Hepatic Fat Accumulation (HFA)', 'Total Cholesterol (TC)']])

# Sort only numeric columns by mean value
numeric_cols = gallstone_dataset.select_dtypes(include=[np.number])
mean_sorted = numeric_cols.mean().sort_values(ascending=False)
print(mean_sorted)

# Sort based on gallstone status then vitamin D
sorted_gall_vit = gallstone_dataset.sort_values(by=['Gallstone Status', 'Vitamin D'], ascending=[True, False])
print(sorted_gall_vit[['Gallstone Status', 'Vitamin D']])

# Sort by ALT, showing top 10
ALT_sort = gallstone_dataset.sort_values(by='ALT', ascending=False).head(10)
print(ALT_sort[["Gender", 'ALT']])

# Sort in place by Obesity (%)
obesity_sorting = gallstone_dataset.sort_values(by='Obesity (%)', ascending=False)
print(obesity_sorting[['Obesity (%)']].head(50))

# Sort patients with high obesity and low muscle mass
high_obesity = gallstone_dataset[gallstone_dataset['Obesity (%)'] > 50]
obesity_muscle_sort = high_obesity.sort_values(by='Muscle Mass (MM)', ascending=True)
print(obesity_muscle_sort[['Obesity (%)', 'Muscle Mass (MM)']].head(10))

# Group by gender and calculate mean cholesterol
grouping_1 = gallstone_dataset.groupby('Gender')['Total Cholesterol (TC)'].mean()
print(grouping_1)

# Group by HFA and count number of patients
grouping_2 = gallstone_dataset.groupby('Hepatic Fat Accumulation (HFA)').size()
print(grouping_2)

# Group by gallstone status and compute median BMI
grouping_3 = gallstone_dataset.groupby('Gallstone Status')['Body Mass Index (BMI)'].median()
print(grouping_3)

# Group by Gender and HFA and average ALT levels
grouping_4 = gallstone_dataset.groupby(['Gender', 'Hepatic Fat Accumulation (HFA)'])['ALT'].mean()
print(grouping_4)

# Find the most common HFA grade per gender
common_HFA = gallstone_dataset.groupby('Gender')['Hepatic Fat Accumulation (HFA)'].agg(lambda x: x.mode()[0])
print(common_HFA) # x.mode() → returns a Series of most frequent values and [0] → selects the first mode in case there are multiple.

# Aggregate by Age decade and average VFR
gallstone_dataset['Age decade'] = (gallstone_dataset['Age'] // 10) * 10
age_dec_vfr = gallstone_dataset.groupby('Age decade')['Visceral Fat Rating (VFR)'].mean()
print(age_dec_vfr)

# Group by Comorbidity level and count gallstones
# Within each comorbidity subgroup, it counts how many patients have each Gallstone Status value
com_gall_count = gallstone_dataset.groupby('Comorbidity')['Gallstone Status'].value_counts().unstack(fill_value=0) # Converts the second index level (Gallstone Status) into separate columns,
# fill_value=0 replaces missing combinations with 0.
print(com_gall_count)

# Find range of glucose levels for each HFA grade
glucose_range = gallstone_dataset.groupby('Hepatic Fat Accumulation (HFA)')['Glucose'].agg(['min', 'max'])
print(glucose_range)

# Group by both Gender and Diabetes and mean HDL
group_hdl = gallstone_dataset.groupby(['Gender', 'Diabetes Mellitus (DM)'])['High Density Lipoprotein (HDL)'].mean()
print(group_hdl)

# Pivot table with index=Gender, columns=HFA, values=Obesity(%)
pivot_tab = gallstone_dataset.pivot_table(index='Gender', columns='Hepatic Fat Accumulation (HFA)', values='Obesity (%)', aggfunc='mean', fill_value=0)
print(pivot_tab)

# Create a new column: age category (e.g., <40, 40-60, >60 )
def age_category(age):
    if age < 40:
        return '<40'
    elif 40 <= age <= 60:
        return '40-60'
    else:
        return '>60'
    
gallstone_dataset['Age category'] = gallstone_dataset['Age'].apply(age_category)
print(gallstone_dataset[['Age', 'Age category']].head())

# Create BMI category column using pd.cut
bmi_bins = [0, 18.5, 24.9, 29.9, np.inf]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
gallstone_dataset['BMI category'] = pd.cut(gallstone_dataset['Body Mass Index (BMI)'], bins=bmi_bins, labels=bmi_labels)
print(gallstone_dataset[['Body Mass Index (BMI)', 'BMI category']].head(50))

# Standardize cholesterol values (z-score)
gallstone_dataset['Cholesterol z-score'] = (gallstone_dataset['Total Cholesterol (TC)'] - gallstone_dataset['Total Cholesterol (TC)'].mean()) / gallstone_dataset['Total Cholesterol (TC)'].std()
print(gallstone_dataset[['Total Cholesterol (TC)', 'Cholesterol z-score']].head())

# Normalize ALT to range 0-1
gallstone_dataset['ALT norm'] = (gallstone_dataset['ALT'] - gallstone_dataset['ALT'].min()) / (gallstone_dataset['ALT'].max() - gallstone_dataset['ALT'].min())
print(gallstone_dataset[['ALT', 'ALT norm']].head())

# Apply log transform to Triglyceride
gallstone_dataset['Log transform Triglyceride'] = np.log1p(gallstone_dataset['Triglyceride']) # log1p is used to handle zero values
print(gallstone_dataset[['Triglyceride', 'Log transform Triglyceride']].head())

# Create a new column: fat-to-muscle ratio
gallstone_dataset['Fat-to-muscle ratio'] = gallstone_dataset['Total Fat Content (TFC)'] / gallstone_dataset['Muscle Mass (MM)']
print(gallstone_dataset[['Total Fat Content (TFC)', 'Muscle Mass (MM)', 'Fat-to-muscle ratio']].head())

# Calculate BMI from Height and Weight and compare
gallstone_dataset['Calculated BMI'] = gallstone_dataset['Weight'] / (gallstone_dataset['Height'] /100) ** 2
print(gallstone_dataset[['Body Mass Index (BMI)', 'Calculated BMI']].head())

# Flag patients with low hemoglobin (<12)
gallstone_dataset['Low HGB'] = gallstone_dataset['Hemoglobin (HGB)'] < 12
print(gallstone_dataset[['Hemoglobin (HGB)', 'Low HGB']].head(50))

# Calculate average lab values per patient and add it as a column (not biologically meaningful, just for practice)
lab_columns = ['Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)',
               'Triglyceride', 'AST', 'ALT', 'Alkaline Phosphatase (ALP)', 'Creatinine', 'CRP', 'Hemoglobin (HGB)', 'Vitamin D']
gallstone_dataset['Average lab values'] = gallstone_dataset[lab_columns].mean(axis=1)
print(gallstone_dataset[['Average lab values']].head(50))

# Concatenate two subsets of the data 
demographics = gallstone_dataset[['Age', 'Gender', 'Comorbidity']]
labs = gallstone_dataset[['Glucose', 'Total Cholesterol (TC)', 'CRP']]

merged = pd.concat([demographics, labs], axis=1)  # merge side by side
print(merged.head())

## Merge with region data
# Merge dataframe with a new one containing region info
region_data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Avg Income': [55000, 48000, 52000, 60000]
})
# Assign regions randomly for demonstration
gallstone_dataset['Region'] = np.random.choice(region_data['Region'], size=len(gallstone_dataset))
merged_region = pd.merge(gallstone_dataset, region_data, on='Region', how='left')
print(merged_region[['Region', 'Avg Income']].head())   
print(merged_region.head())

# Join lab results to demographics using index
demographics.set_index('Age', inplace=True)
labs.set_index(demographics.index, inplace=True)
joined = demographics.join(labs, how='inner')
print(joined.head())

# Append a new row with average values to the end (append method removed from pandas 2.0)
average_row = gallstone_dataset.mean(numeric_only=True)
gallstone_plus_row = pd.concat([gallstone_dataset, pd.DataFrame([average_row])], ignore_index=True)
print(gallstone_plus_row.tail())

# Simulate a merge with 'how=outer' using Gender and Age
labs_new = gallstone_dataset[['Age', 'Gender', 'Glucose', 'Total Cholesterol (TC)', 'CRP']]
demographics_reset = demographics.reset_index()
outer_merge = pd.merge(demographics_reset, labs_new, on=['Age', 'Gender'], how='outer')
print(outer_merge.head())

# Value counts of Comorbidity levels
value_counts_com = gallstone_dataset['Comorbidity'].value_counts()
print(value_counts_com)

# Frequency of each HFA grade
# option 1
hfa_range_1 = gallstone_dataset['Hepatic Fat Accumulation (HFA)'].value_counts()
print(hfa_range_1)
# option 2
hfa_range_2 = gallstone_dataset.groupby('Hepatic Fat Accumulation (HFA)').size()
print(hfa_range_2)
# option 3 (less elegant)
hfa_range_3 = gallstone_dataset.groupby('Hepatic Fat Accumulation (HFA)')['Gender'].count()
print(hfa_range_3)

# Percentage of Gallstone positive by Gender
gall_pos = gallstone_dataset[gallstone_dataset['Gallstone Status'] == 1]
gall_percent = gall_pos['Gender'].value_counts(normalize=True) * 100
print(gall_percent)

# Crosstab of Gender and HFA
crosstab = pd.crosstab(gallstone_dataset['Gender'], gallstone_dataset['Hepatic Fat Accumulation (HFA)'], 
                       margins=True, normalize='index') # margins=True adds totals, normalize='index' gives row percentages

print(crosstab)

# Top 3 most frequent (ECF/TBW) and (VMA) values
print(gallstone_dataset[['Extracellular Fluid/Total Body Water (ECF/TBW)', 'Visceral Muscle Area (VMA) (Kg)']])
top_ecf = gallstone_dataset['Extracellular Fluid/Total Body Water (ECF/TBW)'].value_counts().nlargest(3)
top_vma = gallstone_dataset['Visceral Muscle Area (VMA) (Kg)'].value_counts().nlargest(3)
print(top_ecf)
print(top_vma)

# Distribution of BMI categories
fig, ax = plt.subplots()
gallstone_dataset['BMI category'].value_counts(normalize=True).plot(kind='bar', ax=ax)
ax.set_title('Distribution of BMI Categories')
plt.xticks(rotation=360)
plt.show()
plt.close(fig)

# Proportion of patients without Hypothyroidism and Hyperlipidemia
hypothyroidism_and_hyperlipidemia = gallstone_dataset[(gallstone_dataset['Hypothyroidism'] == 0) &
                                              (gallstone_dataset['Hyperlipidemia'] == 0)]
print(hypothyroidism_and_hyperlipidemia)
print(hypothyroidism_and_hyperlipidemia.shape[0] / gallstone_dataset.shape[0] * 100)

# Histogram bin frequencies of Low Density Lipoprotein (LDL)
fig, ax = plt.subplots()
gallstone_dataset['Low Density Lipoprotein (LDL)'].plot(kind='hist', figsize=(12,6), ax=ax)
ax.set_title('Histogram of LDL levels')
plt.show()
plt.close(fig)

# Frequency table of GFR in quartiles
gfr_quartiles = pd.qcut(gallstone_dataset['Glomerular Filtration Rate (GFR)'], q=4)
print(gfr_quartiles)
freq_table = gfr_quartiles.value_counts().sort_index()  # sort_index to order Q1→Q4
print(freq_table)

# Value counts of rounded Vitamin D
vit_D_round = gallstone_dataset['Vitamin D'].round()
print(vit_D_round.value_counts().sort_index(ascending=False))

# Use .loc to select male patients with Lean Mass (LM) (%) less than 70%
male_lean_mass = gallstone_dataset.loc[(gallstone_dataset['Gender'] == 'Male') &
                                       (gallstone_dataset['Lean Mass (LM) (%)'] < 70)]
print(male_lean_mass)
print(male_lean_mass[['Gender', 'Lean Mass (LM) (%)']])

# Use .iloc to select specific row/column blocks
print(male_lean_mass.iloc[0:5, 0:5]) # first 5 rows and first 5 columns
print(male_lean_mass.iloc[:, 0:5]) # all rows and first 5 columns

# Melt dataframe to long format for plotting
# In simple words reshape the DataFrame (male_lean_mass for example) from wide format
# (many columns) to long format (fewer columns, more rows)
male_lean_mass_melted = male_lean_mass.melt(var_name='Measurement', value_name='Value')
print(male_lean_mass_melted)

# Use stack/unstack on HFA and Gender counts
hfa_gender_counts = gallstone_dataset.groupby(['Hepatic Fat Accumulation (HFA)', 'Gender']).size()
print(hfa_gender_counts)

hfa_gender_unstacked = hfa_gender_counts.unstack()  # .unstack() takes the inner index level (Gender) and pivots it into columns.
print(hfa_gender_unstacked)

hfa_gender_restacked = hfa_gender_unstacked.stack()
print(hfa_gender_restacked)  # .stack() is the inverse of .unstack(). It pushes a column index back into the row index, creating a longer Series with a MultiIndex.

# Reindex the dataframe using custom sorted order
custom_order = [4, 3, 2, 1, 0]
hfa_gender_reindexed = hfa_gender_unstacked.reindex(custom_order)
print(hfa_gender_reindexed)

# Create multi-index: Gender, Coronary Artery Disease (CAD)
ix = ['Gender', 'Coronary Artery Disease (CAD)', 'Comorbidity']
counts = gallstone_dataset.groupby(ix).size()
print(counts)  # MultiIndex Series: index=(Gender, CAD), value=count
multi_index_df = counts.unstack(fill_value=0)  # Convert inner index level (CAD) to columns
print(multi_index_df)

# Reshape with multi-index: Gender, Comorbidity, Coronary Artery Disease (CAD)
long = multi_index_df.stack().reset_index(name='Count')
print(long)

# Return cross-section from the Series/DataFrame.
cross_with_xs = multi_index_df.xs('Male')
print(cross_with_xs)

# Split dataframe into 2: high vs low ALT
# option 1
median_alt = gallstone_dataset['ALT'].median() 
high_alt = gallstone_dataset[gallstone_dataset['ALT'] > median_alt]
low_alt = gallstone_dataset[gallstone_dataset['ALT'] <= median_alt]
print(high_alt)
print(low_alt)

# option 2 
def alt_split(df, cutoff):
    return (df[df['ALT'] > cutoff], df[df['ALT'] <= cutoff])

high_alt2, low_alt2 = alt_split(gallstone_dataset, median_alt)
print(high_alt2)
print(low_alt2)

# Create a new dataset from the initial one with .copy()
year_dataset = gallstone_dataset[['Gender', 'Region', 'Comorbidity', 'Gallstone Status']].copy() # create a new dataset

# Add the column to the new dataframe with randomly created years of birth
n = gallstone_dataset.shape[0] # number of rows
random_years = np.random.randint(1945, 2001, size=n)
year_dataset['Year of Birth'] = random_years
print(year_dataset.head())

# Transpose the new dataframe
transposed = year_dataset.transpose()
print(transposed)

# Drop the column and keep it as a variable
dropped_col = gallstone_dataset.pop(item='Region')
print(dropped_col)

# Access a single value for a row/column label pair
print(year_dataset.at[3, 'Region'])

# Practice .where and .mask
cond_1 = year_dataset['Comorbidity'] >= 1
where_ex = year_dataset.where(cond_1) # What .where does keeps values where condition is True, replaces the rest with NaN.
mask_ex = year_dataset.mask(cond_1) # What .mask does - the opposite: replaces values where condition is True, keeps the rest.
print(where_ex)
print(mask_ex)

# Seaborn practice
# Plot distribution of Body Mass Index (BMI) using histplot
fig, ax = plt.subplots()
sns.histplot(gallstone_dataset['Body Mass Index (BMI)'], bins=30, kde=True, ax=ax)
ax.set_title('Distribution of Body Mass Index (BMI)')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Frequency')
plt.show()
plt.close(fig)

# Boxplot of ALT by Gender
fig, ax = plt.subplots()
sns.boxplot(data=gallstone_dataset, x='Gender', y='ALT', ax=ax)
ax.set_title('Boxplot of ALT by Gender')
plt.show()
plt.close(fig)

# Violin plot of HDL across HFA grades
fig, ax = plt.subplots()
sns.violinplot(data=gallstone_dataset, x='Hepatic Fat Accumulation (HFA)', y='High Density Lipoprotein (HDL)', ax=ax)
ax.set_title('Violin plot of HDL across HFA grades')
#plt.show()
plt.close(fig)

# Swarm plot of CRP by gallstone status
fig, ax = plt.subplots()
sns.swarmplot(data=gallstone_dataset, x='Gallstone Status', y='CRP', size=1, ax=ax)
ax.set_title('Swarm plot of CRP by Gallstone Status')
plt.show()
plt.close(fig)

# Stripplot of Obesity (%) by Comorbidity level
print(gallstone_dataset["Obesity (%)"].head())
fig, ax = plt.subplots()
sns.stripplot(data=gallstone_dataset, x='Comorbidity', y='Obesity (%)', ax=ax, jitter=True)
sns.despine(ax=ax, left=True)
plt.ylim(0, 100)
plt.show()
plt.close(fig)

# Pairplot for selected lab values
lab_values = gallstone_dataset[['AST', 'Total Cholesterol (TC)', 'Triglyceride']]
g = sns.pairplot(lab_values)
g.figure.suptitle('Pairplot of Selected Lab Values', y=1.02)
plt.show()
plt.close(g.figure)

# Countplot of Gallstone cases
fig, ax = plt.subplots()
sns.countplot(data=gallstone_dataset, x='Gallstone Status', ax=ax)
ax.set_title('Countplot of Gallstone Status')
plt.show()
plt.close(fig)  

# Barplot: mean cholesterol by Gender
fig, ax = plt.subplots()
sns.barplot(data=gallstone_dataset, x='Gender', y='Total Cholesterol (TC)', ax=ax, estimator=np.mean)
ax.set_title('Mean Total Cholesterol (TC) by Gender')
plt.show()
plt.close(fig)

# Heatmap of correlation matrix
corr_matrix = gallstone_dataset.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(20, 16), constrained_layout=True)  # Adjust figure size
sns.heatmap(corr_matrix, fmt=".2f", cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix Heatmap')
#plt.tight_layout()
plt.show()
plt.close(fig)

# Boxenplot of Glucose by Diabetes status
fig, ax = plt.subplots()
sns.boxenplot(data=gallstone_dataset, x='Diabetes Mellitus (DM)', y='Glucose', ax=ax)
ax.set_title('Boxenplot of Glucose by Diabetes Status')
plt.show()
plt.close(fig)

# FacetGrid: Histogram of Age by Gender
g = sns.FacetGrid(gallstone_dataset, col="Gender")
g.map_dataframe(sns.histplot, x="Age", bins=20)
plt.show()
plt.close(g.figure)

# Catplot: BMI vs Gender per Gallstone status
g = sns.catplot(data=gallstone_dataset, x='Gender', y='Body Mass Index (BMI)', hue='Gallstone Status', kind='violin')
g.figure.suptitle('BMI vs Gender per Gallstone Status')
plt.show()
plt.close(g.figure)

# Lineplot of creatinin over sorted Age
gallstone_sorted_age = gallstone_dataset.sort_values(by='Age')
fig, ax = plt.subplots()
sns.lineplot(data=gallstone_sorted_age, x='Age', y='Creatinine', errorbar=('ci', 95), ax=ax)
ax.set_title('Lineplot of Creatinine over Age with 95% CI') # ci - confidence interval
plt.show()
plt.close(fig)

# Scatterplot BMI vs Weight colored by Gallstone Status, use different markers for gender
fig, ax = plt.subplots()
sns.scatterplot(data=gallstone_dataset, x='Body Mass Index (BMI)', y='Weight', hue='Gallstone Status', style='Gender', ax=ax, s=60)
fig.suptitle("Scatterplot of BMI vs Weight", fontsize=14)   # main title
ax.set_title("Colored by Gallstone Status", fontsize=11)    # subtitle
plt.tight_layout()
plt.show()
plt.close(fig)  

# KDE plot of GFR by Gender
fig, ax = plt.subplots()
sns.kdeplot(data=gallstone_dataset, x='Glomerular Filtration Rate (GFR)', hue='Gender', ax=ax)
ax.set_title('KDE plot of GFR by Gender')
plt.show()
plt.close(fig)

# Ridgeplot of Hemoglobin (HGB) by 'Coronary Artery Disease (CAD)'
fig, ax = plt.subplots()
sns.kdeplot(data=gallstone_dataset, x="Hemoglobin (HGB)", hue="Coronary Artery Disease (CAD)", ax=ax)
ax.set_title('Ridgeplot of Hemoglobin (HGB) by Coronary Artery Disease (CAD)')
plt.show()
plt.close(fig)

# Facegrid: AST vs ALT by Coronary Artery Disease (CAD) and Gender
g = sns.FacetGrid(gallstone_dataset, col="Coronary Artery Disease (CAD)", hue="Gender")
g.map_dataframe(sns.scatterplot, x="AST", y="ALT")
g.add_legend()
plt.show()
plt.close(g.figure)
 
# Split violinplot of CRP by gender and Gallstone status
fig, ax = plt.subplots()
sns.violinplot(data=gallstone_dataset, x="Gender", y="CRP", hue="Gallstone Status", split=True, ax=ax)
ax.set_title('Split Violinplot of CRP by Gender and Gallstone Status')
plt.show()
plt.close(fig)

# Jointplot: AST vs ALT with regression line
jp = sns.jointplot(data=gallstone_dataset, x="AST", y="ALT", kind="reg", height=8)
jp.figure.suptitle('Jointplot of AST vs ALT with Regression Line', y=1.02)
plt.show()
plt.close(jp.figure)

# Clustermap of scaled lab features
lab_features = gallstone_dataset[['Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)',
                                 'Triglyceride', 'AST', 'ALT', 'Alkaline Phosphatase (ALP)', 'Creatinine', 'CRP', 'Hemoglobin (HGB)', 'Vitamin D']]
lab_features_scaled = (lab_features - lab_features.mean()) / lab_features.std()  # Standardize features
cg = sns.clustermap(lab_features_scaled.dropna(), cmap='viridis', figsize=(10, 10))
plt.show()
plt.close()

# Custom palette for obesity vs muscle mass 
fig, ax = plt.subplots()
custom_palette = {0: 'lightblue', 1: 'lightgreen', 2: 'orange', 3: 'red', 4: 'darkred'}
sns.scatterplot(data=gallstone_dataset, x='Obesity (%)', y='Muscle Mass (MM)', hue='Hepatic Fat Accumulation (HFA)', palette=custom_palette, ax=ax)
ax.set_title('Obesity vs Muscle Mass colored by HFA')
plt.xlim(0, 100)
plt.show()
plt.close(fig)

# Remove outliers before plotting vitamin D distribution, change figure size and aspect ratio, save plot to file
vit_D_no_outliers = gallstone_dataset[gallstone_dataset['Vitamin D'] < 100]
fig, ax = plt.subplots(figsize=(10, 5))  # Wider figure (aspect ratio 2:1 - A higher aspect ratio → wider, a lower one → taller.)
sns.histplot(vit_D_no_outliers['Vitamin D'], bins=30, kde=True, ax=ax)
ax.set_title('Distribution of Vitamin D (without outliers)')
plt.xlabel('Vitamin D')
plt.ylabel('Frequency')
plt.tight_layout()
#plt.savefig('vitamin_D_distribution.png')  # Save the figure to a file
plt.show()
plt.close(fig)

# Plot sorted barplot of average VFR by Hyperlipidemia status
avg_vfr = gallstone_dataset.groupby('Hyperlipidemia')['Visceral Fat Rating (VFR)'].mean().sort_values(ascending=False).reset_index()
fig, ax = plt.subplots()
sns.barplot(data=avg_vfr, x='Hyperlipidemia', y='Visceral Fat Rating (VFR)', ax=ax)
ax.set_title('Average VFR by Hyperlipidemia Status')
plt.show()
plt.close(fig)

# Use swarmplot and boxplot together (violin overlay) for triglyceride by Diabetes status
fig, ax = plt.subplots()
sns.boxplot(data=gallstone_dataset, x='Diabetes Mellitus (DM)', y='Triglyceride', ax=ax, showcaps=False, boxprops={'facecolor':'None'}, whiskerprops={'linewidth':0}, zorder=1)
sns.swarmplot(data=gallstone_dataset, x='Diabetes Mellitus (DM)', y='Triglyceride', ax=ax, color='0.25', size=3, zorder=2)
ax.set_title('Triglyceride by Diabetes Status with Boxplot and Swarmplot')
plt.show()
plt.close(fig)

# Multi-panel plots using col and row arguments
g = sns.catplot(data=gallstone_dataset, x='Gender', y='Body Mass Index (BMI)', hue='Gallstone Status', col='Diabetes Mellitus (DM)', kind='violin', height=5, aspect=1)
g.figure.suptitle('BMI by Gender and Gallstone Status per Diabetes Status', y=1.03)
plt.show()                    
