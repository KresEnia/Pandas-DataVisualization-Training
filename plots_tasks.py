# Load the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataset
liver_data = pd.read_csv('HepatitisCdata.csv')

# Quick look into data
print(liver_data.head(50))

# Data Preprocessing
# Set first column as index without the name
liver_data.set_index(liver_data.columns[0], inplace=True)
liver_data.index.name = None
print(liver_data.head())

# Summary statistics
print(liver_data.describe())
print(liver_data.info())

# Check for missing values
print(liver_data.isnull().sum())

# Add values to missing data if any with mean of the column
for col in liver_data.select_dtypes(include='number').columns:
    liver_data.loc[:, col] = liver_data[col].fillna(liver_data[col].mean())

print(liver_data.isnull().sum())

# Data visualization using Matplotlib
# Bar plot of patient count per category with annotations
plt.figure(figsize=(12, 6))
category_counts = liver_data['Category'].value_counts()
bars = plt.bar(category_counts.index, category_counts.values, color='teal')
plt.title('Patient Count per Category with Annotations')
plt.xlabel('Category')
plt.ylabel('Count')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
plt.show()

# Scatter plot of Age vs CHOL
plt.figure(figsize=(10, 6))
plt.scatter(x = liver_data['Age'], y = liver_data['CHOL'], color='green', alpha=0.5)
plt.title('Age vs CHOL')
plt.xlabel('Age')
plt.ylabel('CHOL')
plt.show()

# Histogram of PROT values
plt.figure(figsize=(10, 6))
plt.hist(liver_data['PROT'], bins=90, facecolor ='#2ab0ff', edgecolor="#24434f", linewidth=0.5)
plt.title('Distribution of PROT values')
plt.xlabel('PROT')
plt.ylabel('Frequency')
plt.show()

# Pie chart of gender distribution
gender_counts = liver_data['Sex'].value_counts()
explode = (0, 0.1)  # explode the 2nd slice- 'Female'
plt.figure(figsize=(8, 8)  )
plt.pie(gender_counts, labels = gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'], explode=explode, shadow=True)
plt.title('Gender Distribution')
plt.show()

# Box plot of GGT grouped by Category
categories = liver_data['Category'].unique() # Get unique categories
data = [liver_data[liver_data['Category'] == cat]['GGT'] for cat in categories] # Create list of GGT arrays, one per category
plt.figure(figsize=(10, 6))
plt.boxplot(data, vert=True, patch_artist=True, labels=categories)
plt.title('Box plot of GGT grouped by Category')
plt.xlabel('Category')
plt.ylabel('GGT')
plt.show()

# Subplots of ALP, ALT, ALP, AST side by side
fig, axs = plt.subplots(1,4, figsize=(20, 5))
axs[0].hist(liver_data['ALP'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axs[0].set_title('ALP Distribution')
axs[0].set_xlabel('ALP')
axs[0].set_ylabel('Frequency')  
axs[1].hist(liver_data['ALT'], bins=30, color='orange', alpha=0.7, edgecolor='black')
axs[1].set_title('ALT Distribution')
axs[1].set_xlabel('ALT') 
axs[2].hist(liver_data['AST'], bins=30, color='green', alpha=0.7, edgecolor='black')
axs[2].set_title('AST Distribution')
axs[2].set_xlabel('AST')
axs[3].hist(liver_data['ALB'], bins=30, color='red', alpha=0.7, edgecolor='black')
axs[3].set_title('ALB Distribution')
axs[3].set_xlabel('ALB')  
plt.tight_layout()
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, bottom=0.15, left=0.1)
plt.show()

# Change markers in a plot, add legend, add grid
plt.figure(figsize=(10, 6))
plt.scatter(liver_data['Age'], liver_data['CHE'], color='blue', marker='x', alpha=0.6, label='CHE Levels')
plt.title('Age vs CHE with Customizations')
plt.xlabel('Age')
plt.ylabel('CHE')
plt.grid(True)
plt.legend()
plt.show()

# Plot error bars for cholesterol between categories
group_stats = liver_data.groupby('Category')['CHOL'].agg(['mean', 'std'])

plt.figure(figsize=(8,6))
plt.errorbar(
    group_stats.index,
    group_stats['mean'],
    yerr=group_stats['std'],
    fmt='o',
    capsize=5,
    ecolor='black',
    color='red'
)

plt.title('Mean CHOL per Category with Standard Deviation')
plt.ylabel('CHOL')
plt.xlabel('Category')
plt.grid(True, alpha=0.3)
plt.show()

# Plot multiple lines in one plot, use different line styles, 
# add legend to top right corner, use twin axes
# 1. Aggregate by Age: mean CHOL and mean GGT per age
age_stats = (
    liver_data
    .groupby('Age', as_index=False)[['CHOL', 'GGT']]
    .mean()
    .sort_values('Age')
)
# 2. Plot with twin axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
# CHOL on left y-axis
line1, = ax1.plot(
    age_stats['Age'],
    age_stats['CHOL'],
    'r--',
    label='Mean CHOL'
)
# GGT on right y-axis
line2, = ax2.plot(
    age_stats['Age'],
    age_stats['GGT'],
    'b-.',
    label='Mean GGT'
)
ax1.set_xlabel('Age')
ax1.set_ylabel('CHOL (mean)', color='r')
ax2.set_ylabel('GGT (mean)', color='b')
ax1.set_title('Mean CHOL and GGT by Age (Twin Axes)')
# 3. Single legend in the top right with both lines
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')
plt.tight_layout()
plt.show()

# Create step plot of cumulative number of patients by age
plt.figure(figsize=(10, 6))
plt.step(
    liver_data['Age'].sort_values(),
    range(len(liver_data)),
    where='post'
)
plt.title('Step Plot of Age')
plt.xlabel('Age')
plt.ylabel('Cumulative Count')
plt.grid(True)
plt.show()

# Example of fill area between two lines and add of axhline with a plot of 
# CHOL vs Age with Highlighted Abnormal Cholesterol Levels
# 1. Compute mean CHOL per Age
age_chol = (
    liver_data
    .groupby('Age', as_index=False)['CHOL']
    .mean()
    .sort_values('Age')
)
x = age_chol['Age']
y = age_chol['CHOL']
normal_limit = 5.0  # mmol/L

# 2. Plot
plt.figure(figsize=(12, 6))
plt.plot(x, y, color='blue', marker='o', label='Mean CHOL per Age')

# 3. Fill area where mean CHOL > 5
plt.fill_between(
    x, y, normal_limit,
    where=(y > normal_limit),
    alpha=0.3,
    color='red',
    label='Mean CHOL > 5 mmol/L'
)

# 4. Horizontal normal limit
plt.axhline(
    y=normal_limit,
    color='black',
    linestyle='--',
    linewidth=1.2,
    label='Normal Limit (5.0 mmol/L)'
)
plt.xlabel('Age (years)')
plt.ylabel('Cholesterol (mmol/L)')
plt.title('Mean CHOL by Age with Highlighted High Values')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Use log scale on GGT axis and zoom into subregion of plot
df_1 = liver_data.groupby('Age', as_index=False)['GGT'].mean().sort_values('Age')

plt.figure(figsize=(12, 6))
plt.plot(df_1['Age'], df_1['GGT'], marker='o', linestyle='-', alpha=0.7)
plt.yscale('log')  # log scale for GGT
plt.xlabel('Age')
plt.ylabel('GGT (log scale)')
plt.title('GGT vs Age (Log Scale)')
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.ylim(20, 80)   # zoom into normal range
plt.xlim(30, 50)   # zoom into specific age region
plt.show()

# Plot cumulative histogram for GGT values + change font sizes in a plot
plt.figure(figsize=(10, 6))
plt.hist(liver_data['GGT'], bins=50, cumulative=True, color='purple', edgecolor='black', alpha=0.7)
plt.title('Cumulative Histogram of GGT', fontsize=16)
plt.xlabel('GGT', fontsize=14)
plt.ylabel('Cumulative Frequency', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# A little bit of explanations for cumulative histogram: 
# A cumulative histogram simply draws:
#x-axis = GGT values
#y-axis = how many people you’ve counted so far

#Plot hexbin of ALB vs PROT + hide the top and right spines
x = liver_data['PROT']
y = liver_data['ALB']
#plt.style.use('seaborn-v0_8-darkgrid') could be used for customize plot style (or classic, ggplot, seaborn-whitegrid, etc)
fig, ax = plt.subplots(figsize=(10,6))
hb = ax.hexbin(x, y, gridsize=50, cmap='Purples', mincnt=1)
cb = plt.colorbar(hb, ax=ax)
cb.set_label('Counts in bin')
ax.set_xlabel('PROT')
ax.set_ylabel('ALB')
ax.set_title('Hexbin plot of PROT vs ALB')
# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# Some interpretation of hexbin plot:
# Darker hexagons (deep purple) = many patients fall in this PROT–ALB range
# Light purple = few patients
# White hexagons = almost no patients
# What the plot reveals about the dataset
# 1. Most patients cluster around:
# Protein 65–80 g/L
# Albumin 35–50 g/L
# This is a typical normal-to-mildly-abnormal range.
# 2. Albumin and protein are positively correlated
# Meaning:
# Higher protein tends to go with higher albumin
# Lower protein tends to go with lower albumin
# This is expected physiologically, since albumin is the major serum protein.

# Plotly Tasks
# Interactive scatterplot of Age vs CHE by gender
fig = px.scatter(
    liver_data,
    x='Age',
    y='CHE',
    color='Sex',
    title='Interactive Scatterplot of Age vs CHE by Gender'
)
fig.show()

# Interactive bar plot of mean CREA per category, use color_discrete_map for palette control
mean_crea = liver_data.groupby('Category', as_index=False)['CREA'].mean()
fig = px.bar(mean_crea, x='Category', y='CREA', title='Mean CREA per Category', color='Category', color_discrete_map={'0=Blood Donor': 'blue', '0s=suspect Blood Donor': 'cyan', '1=Hepatitis': 'green', '2=Fibrosis': 'red', '3=Cirrhosis': 'orange'})
fig.show()

# Interactive pie chart of age group distributions
age_groups = pd.cut(liver_data['Age'], bins=[0, 20, 40, 60, 80], labels=['0-20', '21-40', '41-60', '61-80'])
age_group_counts = age_groups.value_counts().reset_index()
age_group_counts.columns = ['Age Group', 'Count']
fig = px.pie(age_group_counts, names='Age Group', values='Count', title='Age Group Distribution')
fig.show()

# Sunburst chart of Category by gender
fig = px.sunburst(
    liver_data,
    path=['Category', 'Sex'],
    title='Sunburst Chart of Category by Gender'
)
fig.show()

# Treemap of Category > Sex > CHOL
fig = px.treemap(
    liver_data,
    path=['Category', 'Sex'],
    values='CHOL',
    title='Treemap of Category > Sex > CHOL'
)
fig.show()

# Box plot of ALT by Category using Plotly
fig = px.box(liver_data, x='Category', y='ALT', title='Box Plot of ALT by Category')
fig.show()

# Histogram of BIL values with sliders to filter by Age
fig = px.histogram(
    liver_data,
    x='BIL',
    nbins=50,
    title='Histogram of BIL values with Age Filter',
    animation_frame=pd.cut(liver_data['Age'], bins=[0, 20, 40, 60, 80], labels=['0-20', '21-40', '41-60', '61-80'])
)
fig.show()

# Violin plot of PROT by Gender
fig = px.violin(liver_data, x='Sex', y='PROT', box=True, title='Violin Plot of PROT by Gender')
fig.show()

# Make grouped bar plots of mean ALB by category and gender
mean_alb = liver_data.groupby(['Category', 'Sex'], as_index=False)['ALB'].mean()
fig = px.bar(
    mean_alb,
    x='Category',
    y='ALB',
    color='Sex',
    barmode='group',
    title='Mean ALB by Category and Gender'
)
fig.show()  

# 3d scatter plot of ALB, PROT, and CREA
fig = px.scatter_3d(
    liver_data,
    x='ALB',
    y='PROT',
    z='CREA',
    color='Category',
    title='3D Scatter Plot of ALB, PROT, and CREA'
)
fig.show()

# Polar plot for multi-lab values 
lab_means = liver_data[['ALT', 'AST', 'CHOL', 'GGT', 'BIL']].mean().reset_index()
lab_means.columns = ['Lab Test', 'Mean Value']
fig = px.line_polar(
    lab_means,
    r='Mean Value',
    theta='Lab Test',
    line_close=True,
    title='Polar Plot of Mean Lab Values'
)
fig.show()

# Plot heatmap of lab correlation matrix
lab_corr = liver_data.select_dtypes(include='number').corr()
fig = px.imshow(lab_corr, text_auto=True, title='Heatmap of Lab Correlation Matrix')
fig.show()