# Created by s224349, s223289, s237034.

# This program aims to visualize our dataset and carry out
# a PCA analysis of the data, using libraries from the
# exercises done throughout the course, and Seaborn.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# read in the dataset
url = "https://hastie.su.domains/ElemStatLearn/datasets/LAozone.data"
data = pd.read_csv(url)

# list of variables for visualization (without 'doy')
variables = ['ozone', 'vh', 'wind', 'humidity', 'temp', 'ibh', 'dpg', 'ibt', 'vis']

# create boxplots for each of the variables
for variable in variables:
    plt.figure(figsize=(5, 2))
    sns.boxplot(x=data[variable])
    plt.title(f"Boxplot for {variable}")
    plt.show()

# create histograms for each of the variables
for i in range(0, 9, 3):
    fig, axs = plt.subplots(nrows=3, figsize=(7, 7))
    for j in range(3):
        ax = axs[j]
        idx = i + j 
        sns.histplot(data[variables[idx]], kde=True, ax=ax) # histogram with added kde line
        ax.set_title(f"Distribution of {variables[idx]}")
    plt.tight_layout()
    plt.show()

# correlation heatmap/matrix
plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Principal Component Analysis / PCA
# Standardizing the data
scaled_data = StandardScaler().fit_transform(data)

# apply PCA to dataset
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# variables for percentage of variance and labels
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# construct the scree plot
plt.bar(x=range(1,len(per_var)+1), height=per_var)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# Scree plot of cumulative variance with number of principal components
cumulative_var = np.cumsum(per_var)

fig, ax = plt.subplots()
ax.bar(x=range(1, len(per_var) + 1), height=per_var)
ax.plot(range(1, len(cumulative_var) + 1), cumulative_var, color='red', marker='o')

ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Percentage Explained Variance')
plt.title('Cumulative Explained Variance')
plt.show()


# Create the PCA scatter plot
pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df['Ozone'] = data['ozone']  # 'ozone' is target variable

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Ozone', data=pca_df, palette='coolwarm')
plt.title('PCA Graph Colored by Ozone Levels')
plt.xlabel(f'PC1 - {per_var[0]}%')
plt.ylabel(f'PC2 - {per_var[1]}%')
plt.show()

# Create a biplot
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='grey', alpha=0.4)
for i, var in enumerate(data):
    plt.arrow(0, 0, pca.components_[0, i]*max(pca_data[:, 0]), pca.components_[1, i]*max(pca_data[:, 1]), color='red')
    plt.text(pca.components_[0, i]*max(pca_data[:, 0]), pca.components_[1, i]*max(pca_data[:, 1]), var, color='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Biplot of PCA')
plt.grid(True)
plt.show()

# General statistics
cols = range(0, 10)
df_selected = data.iloc[:, cols]
means = df_selected.mean()
medians = df_selected.median()
modes = df_selected.mode().iloc[0]
stdevs = df_selected.std()

# Print the statistics
print("Statistical Summary:\n")
print(f"{'Column':<15}{'Mean':<15}{'Median':<15}{'Mode':<15}{'Std Dev':<15}")
print("-" * 70)

for column in df_selected.columns:
    print(f"{column:<15}{means[column]:<15.2f}{medians[column]:<15.2f}{modes[column]:<15.2f}{stdevs[column]:<15.2f}")