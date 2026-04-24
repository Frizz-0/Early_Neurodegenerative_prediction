import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("oasis_longitudinal_demographics.xlsx")

# Style
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 1. Age vs Brain Volume
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Age', y='nWBV', hue='Group')
plt.title('Brain Volume vs Age by Group')
plt.show()

# ---------------------------------------------------------
# 2. MMSE vs Group
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Group', y='MMSE')
plt.title('MMSE vs Dementia Group')
plt.show()

# ---------------------------------------------------------
# 3. Age Distribution
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# ---------------------------------------------------------
# 4. Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# ---------------------------------------------------------
# 5. Longitudinal Analysis (Patient History)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))

for subject in df['Subject ID'].unique()[:5]:
    subj_data = df[df['Subject ID'] == subject].sort_values('Visit')
    plt.plot(subj_data['Visit'], subj_data['nWBV'], marker='o', label=subject)

plt.title('Patient Brain Volume Change Over Time')
plt.xlabel('Visit Number')
plt.ylabel('nWBV')
plt.legend()
plt.show()

# ---------------------------------------------------------
# 6. Dementia Progression (Converted Patients)
# ---------------------------------------------------------
converters = df[df['Group'] == 'Converted']['Subject ID'].unique()[:5]

plt.figure(figsize=(10,6))

for subject in converters:
    subj_data = df[df['Subject ID'] == subject].sort_values('Visit')
    plt.plot(subj_data['Visit'], subj_data['nWBV'], marker='o', label=subject)

plt.title('Brain Volume Decline in Converted Patients')
plt.xlabel('Visit')
plt.ylabel('nWBV')
plt.legend()
plt.show()

# ---------------------------------------------------------
# 7. Regression Analysis (Normal vs Dementia)
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

sns.regplot(
    data=df[df['Group']=='Nondemented'],
    x='Age', y='nWBV',
    label='Nondemented'
)

sns.regplot(
    data=df[df['Group']=='Demented'],
    x='Age', y='nWBV',
    color='red',
    label='Demented'
)

plt.title('Brain Atrophy Comparison')
plt.legend()
plt.show()

# ---------------------------------------------------------
# 8. Summary Statistics
# ---------------------------------------------------------
print("Summary Statistics:")
print(df.groupby('Group')[['Age','MMSE','nWBV']].mean())