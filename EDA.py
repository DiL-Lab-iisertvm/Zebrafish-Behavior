# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
# load dataset
data = pd.read_csv('D:/IISER TVM/Projects/Data/NTD_Acute_Stress.csv')
##################################################
############summary statistics ############
# data = data.drop('NAME' , axis = 1)
# data = data.drop('VERDICT' , axis = 1)
# summary_stats = data.describe().T
# summary_stats['mean'] = data.mean()
# summary_stats['median'] = data.median()
# summary_stats['std'] = data.std()
# # Rearranging and renaming columns for clarity
# summary_stats = summary_stats[['mean', '50%', 'std', 'min', 'max']]
# summary_stats.columns = ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max']
# # Plotting
# fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
# # List of statistics
# #stats = ['Mean', 'Median', 'Standard Deviation']
# stats = ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max']
# # Plot each statistic
# for i, stat in enumerate(stats):
#     axes[i].bar(summary_stats.index, summary_stats[stat], color='skyblue')
#     axes[i].set_title(stat)
#     axes[i].set_xlabel('Features')
#     if i == 0:
#         axes[i].set_ylabel('Value')
# plt.tight_layout()
# plt.show()
###################################################
import missingno as msno
#data = data.drop('NAME' , axis = 1)
msno.bar(data , figsize = (16,5),color = "red")
plt.savefig("null_values.png")
plt.show()
############################################
#histogram
# num_rows = ["TOTAL_ID", "TIME_SIB", "TIME_SIT"]
# for num_row in num_rows:
#     sns.histplot(data=data, x=num_row, hue='VERDICT', multiple="stack", palette="tab10")
#     plt.title(f"{num_row} Histogram")
#     plt.xlabel(f"{num_row}")
#     plt.ylabel("Frequency")
#     plt.show()

#####################################################
#correlation map
corr = data.corr()
plt.figure(figsize=[6,6])
sns.heatmap(corr,cmap=sns.cubehelix_palette(as_cmap=True),annot=True)
plt.show()
######################
#BoxPlot
#plt.figure(figsize=(15, 10))

#TOTAL_ID box plot
# plt.subplot(3, 1, 1)
# sns.boxplot(x='VERDICT', y='TOTAL_ID', data=data, palette='pastel')
# plt.title('Box Plot of TOTAL_ID by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TOTAL_ID')
# plt.show()
# # TIME_SIB box plot
# plt.subplot(3, 1, 2)
# sns.boxplot(x='VERDICT', y='TIME_SIB', data=data, palette='pastel')
# plt.title('Box Plot of TIME_SIB by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TIME_SIB')
# plt.show()
#
# # TIME_SIT box plot
# plt.subplot(3, 1, 3)
# sns.boxplot(x='VERDICT', y='TIME_SIT', data=data, palette='pastel')
# plt.title('Box Plot of TIME_SIT by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TIME_SIT')
# plt.show()

# Adjust layout
# plt.tight_layout()
# plt.show()
##################################################
#boxplot
# data = data.drop('VERDICT' , axis = 1)
# sns.boxplot(data)
# plt.ylabel("Value")
# plt.show()
#violin plots
# num_features = data.select_dtypes(include=['number']).columns
# # Create a subplot for each numerical feature
# fig, axes = plt.subplots(len(num_features), 1, figsize=(10, 6 * len(num_features)))
# for i, feature in enumerate(num_features):
#     ax = axes[i] if len(num_features) > 1 else axes
#     sns.violinplot(y=feature, data=data, ax=ax)
#     ax.set_title(f'Violin Plot of {feature}')
#     ax.set_xlabel('Density')
#     ax.set_ylabel(feature)
# plt.tight_layout()
# plt.show()
#############
#plt.figure(figsize=(15, 10))

# TOTAL_ID violin plot
#plt.subplot(3, 1, 1)
# sns.violinplot(x='VERDICT', y='TOTAL_ID', data=data, palette='pastel')
# plt.title('Violin Plot of TOTAL_ID by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TOTAL_ID')
# plt.show()

# TIME_SIB violin plot
#plt.subplot(3, 1, 2)
# sns.violinplot(x='VERDICT', y='TIME_SIB', data=data, palette='pastel')
# plt.title('Violin Plot of TIME_SIB by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TIME_SIB')
# plt.show()
#
# # TIME_SIT violin plot
# plt.subplot(3, 1, 3)
# sns.violinplot(x='VERDICT', y='TIME_SIT', data=data, palette='pastel')
# plt.title('Violin Plot of TIME_SIT by Anxiety Verdict')
# plt.xlabel('Anxiety Verdict')
# plt.ylabel('TIME_SIT')
# plt.show()
# Adjust layout
# plt.tight_layout()
# plt.show()

#################################
# sns.pairplot(data, hue="VERDICT", markers=["o", "s"])
# plt.show()
#####################################
#PCA
from sklearn.decomposition import PCA
X = data.drop('NAME', axis=1)
y = data['VERDICT']
X = data.drop('VERDICT', axis=1)
X = data.drop('NAME', axis=1)
print(X)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['VERDICT'] = y
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
# Plot the explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid', linestyle='--', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.legend(loc='best')
plt.show()
# Plotting the principal components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='VERDICT', data=pca_df)
plt.title('PCA of Zebrafish Anxiety Behaviors Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='VERDICT')
plt.grid(True)
plt.show()
# Display the PCA DataFrame
print(pca_df)

###############################
X = data.drop('VERDICT' , axis = 1)
y = data['VERDICT']
value_counts = y.value_counts()
print("Count of 0:", value_counts[0])
print("Count of 1:", value_counts[1])
sns.countplot(x=y)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Countplot of VERDICT')
plt.show()
############################
plt.figure(figsize = (15,5))
colors = ['darkorange','royalblue']
my_explode = [0,0.1]
my_labels = ['Anxious', 'Non-Anxious']
data['VERDICT'].value_counts().plot(kind = 'pie',autopct = '%.2f%%',explode= my_explode, colors=colors, labels= my_labels)
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#swarmPlot
#Visualize how different features impact the target variable (Anxiety Level)
# df = data.drop('VERDICT', axis=1)
# df = data.drop('NAME', axis=1)
# features = ['TOTAL_ID', 'TIME_SIB', 'TIME_SIT']
# plt.figure(figsize=(15, 5))
# for i, feature in enumerate(features, 1):
#     plt.subplot(1, len(features), i)
#     sns.swarmplot(x='VERDICT', y=feature, data=df)
#     plt.title(f'Swarm Plot of {feature} by Anxiety Level')
# plt.tight_layout()
# plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
