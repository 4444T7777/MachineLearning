# Created by s224349, s223289, s237034.

# This program aims to visualize our dataset and carry out
# a PCA analysis of the data, using libraries from the
# exercises done throughout the course, and Seaborn.

# Additional help for the PCA part comes from StatQuests PCA guide:
# https://github.com/StatQuest/pca_demo/blob/master/pca_demo.py

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
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# 2D plot of PC1 and PC2
pca_df = pd.DataFrame(pca_data, index=data.index, columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.show()