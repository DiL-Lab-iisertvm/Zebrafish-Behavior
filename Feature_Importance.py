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
data = pd.read_csv('D:/IISER TVM/Projects/Data/NTD_Controlled_Stress.csv')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features = data.copy()
features.drop(columns = ['NAME','VERDICT'],inplace = True)
print(features)
X=features
label = data['VERDICT']
y=label
##################################
# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(features, label)

# Extract feature importances
feature_importances = rf_classifier.feature_importances_
print(feature_importances)

features_names = ['TOTAL_ID', 'TIME_SIB', 'TIME_SIT', 'Large_angle', 'Small_angle']
# Plot the feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features_names)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Zebrafish Anxiety Prediction')
#plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()
