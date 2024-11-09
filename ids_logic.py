import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model import modelRFC, modelDTC, modelKNN, modelGNB

from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc
import warnings
warnings.filterwarnings('ignore')


np.set_printoptions(precision=3)
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#load the dataset
test_url = 'dataset.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

label_mapping = {
    0: "Normal",
    1: "DoS",
    2: "R2L",
    3: "U2R",
    4: "Probe",
    5: "DoS",
    6: "U2R",
    7: "R2L",
    8: "DoS",
    9: "Probe",
    10: "R2L",
    11: "DoS"
}

def preprocess(dataset):
    new_data = pd.read_csv(dataset, header=None, names=col_names)

    new_data = new_data.drop(columns=["label"])  # Replace "target" with your target column name

    # Encode categorical columns in X if any
    new_data = pd.get_dummies(new_data)

    # Align the columns of new_data to match the trained model's features

    new_data.head(5) 
    new_data = pd.get_dummies(new_data)


    new_data.head(5)
    new_data.columns = new_data.columns.astype(str)

    return new_data
#DECISION TREE CLASSIFIER
def DecisionTree(new_data):
    #DECISION TREE CLASSIFIER
    predictions = modelDTC.predict(new_data)
    print(predictions)
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]
# Create a DataFrame for better visualization
    results_df = pd.DataFrame({
    'Predicted Label': predicted_labels
})

    # Count the occurrences of each label
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    #Assuming `predictions` is your array of predicted labels
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]
    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({
        'Predicted Label': predicted_labels
    })

    # Count the occurrences of each label
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax, title='Distribution of Predictions')
    label_counts.plot(kind='bar', title='Distribution of Predictions')
   
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

#NaiveBayes   
# def NaiveBayes():

#     #NAIVEBAYES
#     clf_Naive = GaussianNB()
#     train0 = time.time()
#     # Train Decision Tree Classifer
#     clf_Naive = clf_Naive.fit(X_Df, Y_Df.astype(int))
#     train1 = time.time() - train0

#     test0 = time.time()
#     Y_Df_pred=clf_Naive.predict(X_Df_test)
#     test1 = time.time() - test0


#     # Create confusion matrix
#     confusion_matrixNaiveBayes = pd.crosstab(Y_Df_test, Y_Df_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])


#     # Save trained model
#     with open('IDS_model_NaiveBayes.pkl', 'wb') as file:
#         pickle.dump(clf_Naive, file)
#     return confusion_matrixNaiveBayes,{'Accuracy': accuracy,'Precision': precision,'Recall': recall,'F-measure': f,'Train':train1, 'Test':test1}
    
# #K-NEAREST NEIGHBOUR
# def KNN():

#     clf_KNN = KNeighborsClassifier()
#     train0 = time.time()
#     clf_KNN.fit(X_Df, Y_Df.astype(int))
#     train1 = time.time() - train0

#     #for the test dataset
#     test0 = time.time()
#     Y_pred = clf_KNN.predict((X_Df_test))
#     test1 = time.time() - test0

#     #confusion matrix
#     confusion_matrixKNN = pd.crosstab(Y_Df_test, Y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])

#     #accuracy, precision, f-measure and train time scores
#     accuracy = cross_val_score(clf_KNN, X_Df_test, Y_Df_test, cv=10, scoring='accuracy')
#     print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
#     precision = cross_val_score(clf_KNN, X_Df_test, Y_Df_test, cv=10, scoring='precision')
#     print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
#     recall = cross_val_score(clf_KNN, X_Df_test, Y_Df_test, cv=10, scoring='recall')
#     print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
#     f = cross_val_score(clf_KNN, X_Df_test, Y_Df_test, cv=10, scoring='f1')
#     print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
#     print("train_time:%.3fs\n" %train1)
#     print("test_time:%.3fs\n" %test1)

#      # Save trained model
#     with open('IDS_model_KNN.pkl', 'wb') as file:
#         pickle.dump(clf_KNN, file)
#     return confusion_matrixKNN,{'Accuracy': accuracy,'Precision': precision,'Recall': recall,'F-measure': f,'Train':train1, 'Test':test1}
