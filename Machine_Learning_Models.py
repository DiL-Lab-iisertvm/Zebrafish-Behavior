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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features = data.copy()
features.drop(columns = ['NAME','VERDICT'],inplace = True)
print(features)
X=features
label = data['VERDICT']
y=label
# X_train, X_test, y_train, y_test =train_test_split(features, label, test_size=0.3, random_state=5)
#####################################
# model = LogisticRegression(C=1, max_iter=300, penalty='l1', solver='liblinear')
# model.fit(X_train, y_train)
# # Making Prediction
# pred = model.predict(X_test)
# print(classification_report(y_test,pred))
# # confusion Maxtrix
# cm1 = confusion_matrix(y_test, pred)
# # Define labels for the confusion matrix
# labels = ['Non-Anxious','Anxious']
# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm1/np.sum(cm1), annot = True, fmt=  '0.2%', cmap = 'Reds',xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# #plt.title('Confusion Matrix')
# plt.show()
##########################################
# Define a list of classifiers/models to compare
models = [
    ('Logistic Regression', LogisticRegression(C=1, max_iter=300, penalty='l1', solver='liblinear')),
    ('SVM', SVC(C=10, gamma='auto', kernel='rbf')),
    ('Gaussian Naive Bayes', GaussianNB(var_smoothing=1e-05)),
    ('Decision Tree', DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None,
                                             min_samples_leaf=2, min_samples_split=5)),
    ('Random Forest', RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)),
    ('k-Nearest Neighbors', KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform'))
]


# Define the number of folds for cross-validation
k = 10
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
# Create a KFold object for K-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Dictionary to store cross-validation results
cross_val_results = {}

# Iterate through each model and perform K-fold cross-validation
for model_name, model in models:
    # Start the timer for cross-validation
    # start_time = time.time()

    # Perform K-fold cross-validation and calculate the mean accuracy
    cross_val_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_accuracy = np.mean(cross_val_scores)
    y_pred = cross_val_predict(model, X, y, cv=kf)


    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    # Print the results for each model along with time taken for cross-validation
    print(f'{model_name}:')
    print(f'Cross-Validation Scores: {cross_val_scores}')
    print(f'Mean Accuracy: {mean_accuracy:.4f}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    confusion_matrix_graph = confusion_matrix(y, y_pred)
    # plt.figure(figsize=(5, 5))
    labels = ['Non-Anxious', 'Anxious']
    sns.heatmap(confusion_matrix_graph /np.sum(confusion_matrix_graph ,axis = 0), annot=True,fmt='.2%', cmap='Blues',xticklabels=labels, yticklabels=labels)
    plt.xticks()
    plt.yticks()
    plt.title(f"Confusion matrix of {model_name} Classifier",fontsize =14)
    plt.xlabel("Predicted",fontsize =12)
    plt.ylabel("Actual",fontsize =12)
    f,ax = plt.subplots()
    f.set_visible(False)
    f.set_figheight(3)
    plt.show()


    # Store cross-validation results
    cross_val_results[model_name] = {
        'Cross-Validation Scores': cross_val_scores,
        'Mean Accuracy': mean_accuracy,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }
     # Save the model

