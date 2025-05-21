#importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#uploading the data set
data = '/Users/dhruvkarande/Documents/Data Analytics Project /Lung Cancer Growth Trends/Lung_Cancer_Trends_Realistic.csv'
df = pd.read_csv(data)
print(df.head())

#Objective of the Project
"""
To analyze and visualize trends in lung cancer incidence and mortality using a realistic dataset, leveraging Python’s core data analysis stack (NumPy, Pandas, and Matplotlib). 
The goal is to extract key insights regarding demographic impacts (such as age and gender), temporal patterns, and regional distributions in order to support data-driven healthcare decisions and awareness initiatives.
"""

# Cleaning the data 
print("Missing values before cleaning:\n", df.isnull().sum())
columns_to_fill = ["Occupation_Exposure", "Alcohol_Consumption", "Lung_Cancer_Stage"]
for col in columns_to_fill:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values after cleaning:\n", df.isnull().sum())

# Selecting specific columns for correlation analysis
specific_data_1 = df[[
    "Years_Smoking", "Secondhand_Smoke_Exposure", "Occupation_Exposure",
    "Air_Pollution_Level", "Family_History", "Genetic_Markers_Positive",
    "Alcohol_Consumption", "Diet_Quality", "Access_to_Healthcare",
    "Chronic_Lung_Disease", "Lung_Cancer_Stage"
]]
print(specific_data_1.head())

# Define columns to visualize
columns_to_plot = [
    "Years_Smoking",
    "Secondhand_Smoke_Exposure",
    "Occupation_Exposure",
    "Air_Pollution_Level",
    "Family_History",
    "Alcohol_Consumption",
    "Diet_Quality"
]

# Function to plot pie charts for a given stage
def plot_stage_pies(df_stage, stage_label):
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    for i, col in enumerate(columns_to_plot):
        counts = df_stage[col].value_counts()
        axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        axes[i].set_title(f"{col.replace('_', ' ')}")
    for j in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(f"Distribution of Various Factors in {stage_label}", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# Plot pie charts for each stage
stages = df['Lung_Cancer_Stage'].unique()
for stage in stages:
    stage_df = df[df['Lung_Cancer_Stage'] == stage]
    plot_stage_pies(stage_df, f"{stage} Lung Cancer")

# Correlation heatmap for numerical and encoded categorical values
numeric_df = specific_data_1.copy()
for col in numeric_df.select_dtypes(include='object').columns:
    numeric_df[col] = numeric_df[col].astype('category').cat.codes
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Heatmap of Risk Factors")
plt.tight_layout()
plt.show()

# Group-wise statistics
print("\nAverage Years Smoking by Lung Cancer Stage:")
print(df.groupby("Lung_Cancer_Stage")["Years_Smoking"].mean())

# Basic insight statements
print("\n🔍 Insights:")
print("- Stage III/IV patients have higher average years of smoking.")
print("- Secondhand smoke and air pollution are widely present across stages.")
print("- Access to healthcare and diet quality show moderate variation across stages.")
