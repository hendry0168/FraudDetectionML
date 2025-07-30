import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

# Load the csv file
dataframe = pd.read_csv(r"source\repos\credit-card-fraud-python-ml-code\creditcard.csv")
# dataframe.head()
# # ### Perform Exploratory Data Analysis
# dataframe.info()
# # Check for null values
# dataframe.isnull().values.any()
# dataframe["Amount"].describe()
non_fraud = len(dataframe[dataframe.Class == 0])
fraud = len(dataframe[dataframe.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100

# print("Number of Genuine transactions: ", non_fraud)
# print("Number of Fraud transactions: ", fraud)
# print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))

# # Visualize the "Labels" column in our dataset
# labels = ["Genuine", "Fraud"]
# count_classes = dataframe.value_counts(dataframe['Class'], sort= True)
# count_classes.plot(kind = "bar", rot = 0)
# plt.title("Visualization of Labels")
# plt.ylabel("Count")
# plt.xticks(range(2), labels)
# plt.show()

# Perform Scaling
scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace= True, axis= 1)

Y = dataframe["Class"]
X = dataframe.drop(["Class"], axis= 1)
# Y.head()

# Split the data
(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.3, random_state= 42)
# print("Shape of train_X: ", train_X.shape)
# print("Shape of test_X: ", test_X.shape)


# Let's train different models on our dataset and observe which algorithm works better for our problem.
# 
# Let's apply Random Forests and Decision Trees algorithms to our dataset.

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)

predictions_dt = decision_tree.predict(test_X)
decision_tree_score = decision_tree.score(test_X, test_Y) * 100

# Random Forest
random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(train_X, train_Y)

predictions_rf = random_forest.predict(test_X)
random_forest_score = random_forest.score(test_X, test_Y) * 100

# xgboost
xgboost = XGBClassifier(n_estimators= 100)
xgboost.fit(train_X, train_Y)

predictions_xg= xgboost.predict(test_X)
xgboost_score = xgboost.score(test_X, test_Y) * 100

# # Print scores of our classifiers
# print("Random Forest Score: ", random_forest_score)
# print("Decision Tree Score: ", decision_tree_score)

# The below function is directly taken from the scikit-learn website to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot confusion matrix for Decision Trees
confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)

plot_confusion_matrix(confusion_matrix_dt, classes=[0, 1], title= "Confusion Matrix - Decision Tree")

# Plot confusion matrix for Random Forests
confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)

plot_confusion_matrix(confusion_matrix_rf, classes=[0, 1], title= "Confusion Matrix - Random Forest")

# Plot confusion matrix for xgboost
confusion_matrix_xg = confusion_matrix(test_Y, predictions_xg.round())
print("Confusion Matrix - xgboost")
print(confusion_matrix_xg)

plot_confusion_matrix(confusion_matrix_xg, classes=[0, 1], title= "Confusion Matrix - xgboost")


# The below function prints the following necesary metrics
def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
    
print("Evaluation of Decision Tree Model")
print()
metrics(test_Y, predictions_dt.round())

print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_rf.round())

print("Evaluation of xgboost Model")
print()
metrics(test_Y, predictions_xg.round())


# Clearly, Random Forest model works better than Decision Trees

# But, if we clearly observe our dataset suffers a serious problem of **class imbalance**. 
# The genuine (not fraud) transactions are more than 99% with the fraud transactions constituting of 0.17%.
# 
# With such kind of distribution, if we train our model without taking care of the imbalance issues, it predicts the label with higher importance given to genuine transactions (as there are more data about them) and hence obtains more accuracy.

# The class imbalance problem can be solved by various techniques. **Over sampling** is one of them.
#  
# One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. 
# 
# Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the **Synthetic Minority Oversampling Technique**, or **SMOTE** for short.

# Performing oversampling on RF and DT
#!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)
value_counts = Counter(Y_resampled)
print(value_counts)
(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)

# Build the Random Forest classifier on the new dataset
rf_resampled = RandomForestClassifier(n_estimators = 100)
rf_resampled.fit(train_X, train_Y)

predictions_resampled = rf_resampled.predict(test_X)
random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100


# Build the xgboost classifier on the new dataset
xg_resampled = XGBClassifier(n_estimators = 100)
xg_resampled.fit(train_X, train_Y)

predictions_resampled_xg = xg_resampled.predict(test_X)
xgboost_score_resampled = xg_resampled.score(test_X, test_Y) * 100


# Visualize the confusion matrix
cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)

plot_confusion_matrix(cm_resampled, classes=[0, 1], title= "Confusion Matrix - Random Forest After Oversampling")
print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_resampled.round())


cm_resampled_xg = confusion_matrix(test_Y, predictions_resampled_xg.round())
print("Confusion Matrix - xgboost")
print(cm_resampled_xg)

plot_confusion_matrix(cm_resampled_xg, classes=[0, 1], title= "Confusion Matrix - xgboost After Oversampling")
print("Evaluation of xgboost Model")
print()
metrics(test_Y, predictions_resampled_xg.round())
