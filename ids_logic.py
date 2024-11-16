import io
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model import modelRFC, modelDTC, modelKNN, modelGNB
from datetime import datetime
from sklearn import preprocessing,metrics

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
    from model import X_columns
    new_data = pd.read_csv(dataset, header=None, names=col_names)

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns = X_columns, fill_value=0)

    return new_data


#DECISION TREE CLASSIFIER
def DecisionTree(new_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    predictions = modelDTC.predict(new_data)
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({'Predicted Label': predicted_labels})
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax, title='Distribution of Predictions')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    filename = f'assets/Confusion Matrices/Confusion Matrices DecisionTree/Plot_DecisionTree({timestamp}).png'
    fig.savefig(filename)

    #Convert plot to base64 for display in HTML
    with open(filename, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return img_base64

def RandomForest(new_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    predictions = modelRFC.predict(new_data)
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({'Predicted Label': predicted_labels})
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax, title='Distribution of Predictions')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    filename = f'assets/Confusion Matrices/Confusion Matrices RandomForest/Plot_RandomForest({timestamp}).png'
    fig.savefig(filename)

    #Convert plot to base64 for display in HTML
    with open(filename, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return img_base64

def KNN(new_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    predictions = modelKNN.predict(new_data)
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({'Predicted Label': predicted_labels})
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax, title='Distribution of Predictions')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    filename = f'assets/Confusion Matrices/Confusion Matrices KNN/Plot_KNN({timestamp}).png'
    fig.savefig(filename)

    #Convert plot to base64 for display in HTML
    with open(filename, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return img_base64

def GaussianNB(new_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    predictions = modelGNB.predict(new_data)
    predicted_labels = [label_mapping.get(pred, "Unknown") for pred in predictions]

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({'Predicted Label': predicted_labels})
    label_counts = results_df['Predicted Label'].value_counts()

    # Plotting the distribution of predictions
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax, title='Distribution of Predictions')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    filename = f'assets/Confusion Matrices/confusion Matrices NaiveBayes/Plot_GaussianNB({timestamp}).png'
    fig.savefig(filename)

    #Convert plot to base64 for display in HTML
    with open(filename, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return img_base64