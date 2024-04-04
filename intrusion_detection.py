import pandas as pd
import numpy as np
import seaborn as sns
import imblearn
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time
import logging

from sklearn import preprocessing,svm,metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc
from sklearn.tree import DecisionTreeClassifier 

import warnings
warnings.filterwarnings('ignore')


np.set_printoptions(precision=3)
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#load the dataset
train_url = 'NSL_KDD/NSL_KDD_Train.csv'
test_url = 'NSL_KDD/NSL_KDD_Test.csv'


categorical_columns=['protocol_type', 'service', 'flag']
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


#load nsl dataset and preprocess it
def load_dataset():
    global confusion_matrix1, confusion_matrix2
    try:
        df = pd.read_csv(train_url, header=None, names=col_names)
        df_test = pd.read_csv(test_url, header=None, names = col_names)
    except FileNotFoundError:
        logging.error(f"File not found: NSL_KDD/NSL_KDD_Train.csv")

    print('Dimensions of the Training set:', df.shape)
    print('Dimensions of the test set: ', df_test.shape)

    df.head()

    print(df.head(10))
    print(df_test.head(10))

    print('Label distribution Training set:')
    print(df['label'].value_counts())
    print()
    print('Label distribution Test set:')
    print(df_test['label'].value_counts())

    #removing the num_outbound_cmds because it is a redundant column
    df.drop(['num_outbound_cmds'], axis=1, inplace=True)
    df_test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    #one-Hot encoding

    #for Training set
    print('Training set:')
    for col_name in df.columns:
     if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    print()
    print('Distribution of categories in service:')
    print(df['service'].value_counts().sort_values(ascending=False).head())
    
    #for test set
    print('Test set:')
    for col_name in df_test.columns:
        if df_test[col_name].dtypes == 'object' :
            unique_cat = len(df_test[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


   #labelEncoder: inserting the categorical features into a 2d numpy array
    df_categorical_values = df[categorical_columns]
    testdf_categorical_values = df_test[categorical_columns]

    df_categorical_values.head()
    print(df_categorical_values.head())
   


   # protocol type
    unique_protocol=sorted(df.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2=[string1 + x for x in unique_protocol]
    print(unique_protocol2)

    # service
    unique_service=sorted(df.service.unique())
    string2 = 'service_'
    unique_service2=[string2 + x for x in unique_service]
    print(unique_service2)


    # flag
    unique_flag=sorted(df.flag.unique())
    string3 = 'flag_'
    unique_flag2=[string3 + x for x in unique_flag]
    print(unique_flag2)


    # put together
    dumcols=unique_protocol2 + unique_service2 + unique_flag2


    #do it for test set
    unique_service_test=sorted(df_test.service.unique())
    unique_service2_test=[string2 + x for x in unique_service_test]
    testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

    #transforming the categorical featurees into numbers using labelEncoders()

    df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

    print(df_categorical_values.head())
    print('--------------------')
    print(df_categorical_values_enc.head())

    # test set
    testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

    #one-Hot Encoding
    enc = OneHotEncoder(categories='auto')
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


    # test set
    testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
    testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

    df_cat_data.head()
    print(df_cat_data.head())


    #Adding missing colums in the dataset
    trainservice=df['service'].tolist()
    testservice= df_test['service'].tolist()
    difference=list(set(trainservice) - set(testservice))
    string = 'service_'
    difference=[string + x for x in difference]
    difference


    for col in difference:
        testdf_cat_data[col] = 0

    print(df_cat_data.shape)    
    print(testdf_cat_data.shape)


    #Adding new numeric columns to mian dataframe
    newdf=df.join(df_cat_data)
    newdf.drop('flag', axis=1, inplace=True)
    newdf.drop('protocol_type', axis=1, inplace=True)
    newdf.drop('service', axis=1, inplace=True)

    # test data
    newdf_test=df_test.join(testdf_cat_data)
    newdf_test.drop('flag', axis=1, inplace=True)
    newdf_test.drop('protocol_type', axis=1, inplace=True)
    newdf_test.drop('service', axis=1, inplace=True)

    print(newdf.shape)
    print(newdf_test.shape)

    #coverting the labels from strings to binary representations
    labeldf=newdf['label']
    labeldf_test=newdf_test['label']


    # change the label column
    newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1,
                            'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1 })
    newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1
                           ,'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1})


    # put the new label column back
    newdf['label'] = newlabeldf
    newdf_test['label'] = newlabeldf_test

    newdf.head()
    print(newdf.head())


    #FEATURE SCALING
    #Splitting dataframes into X and Y
    X_Df = newdf.drop('label',axis=1)
    Y_Df = newdf.label

    # test set
    X_Df_test = newdf_test.drop('label',axis=1)
    Y_Df_test = newdf_test.label

    colNames=list(X_Df)
    colNames_test=list(X_Df_test)

    scaler1 = preprocessing.StandardScaler().fit(X_Df)
    X_Df=scaler1.transform(X_Df) 

    # test data
    scaler5 = preprocessing.StandardScaler().fit(X_Df_test)
    X_Df_test=scaler5.transform(X_Df_test) 


    print(X_Df.shape)
    print(X_Df_test.shape)


    ''' for feature in categorical_columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    print('df[feature]')
    print(df[feature])
    '''



    X_train, X_test, y_train, y_test = train_test_split(X_Df, Y_Df, test_size=0.2, random_state=42)

    ''' # Train RandomForestClassifier mode
    clf = RandomForestClassifier()
    train_forest0=time.time()

    clf.fit(X_Df, Y_Df.astype(int))
    train_forest1 = time.time()-train_forest0

    test_forest0=time.time()
    Y_Df_pred_forest=clf.predict(X_Df_test)
    test_forest1 = time.time() - test_forest0

    # Evaluate model
    y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {accuracy}')
   


    accuracy = cross_val_score(clf, X_Df_test, Y_Df_test, cv=10, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
    precision = cross_val_score(clf, X_Df_test, Y_Df_test, cv=10, scoring='precision')
    print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
    recall = cross_val_score(clf, X_Df_test, Y_Df_test, cv=10, scoring='recall')
    print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
    f = cross_val_score(clf, X_Df_test, Y_Df_test, cv=10, scoring='f1')
    print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
    print("train_time:%.3fs\n" %train_forest1)
    print("test_time:%.3fs\n" %test_forest1)


    # Save trained model
    with open('intrusion_detection_model.pkl', 'wb') as file:
        pickle.dump(clf, file)
'''



    #DECISION TREE CLASSIFIER
    clf_Tree = DecisionTreeClassifier()
    train0=time.time()
    #traning DT
    clf_Tree=clf_Tree.fit(X_Df,Y_Df.astype(int))
    train1=time.time()-train0

    test0=time.time()
    Y_Df_pred=clf_Tree.predict(X_Df_test)
    test1 = time.time() - test0

    # Create confusion matrix
    confusion_matrix1=pd.crosstab(Y_Df_test, Y_Df_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
    print(confusion_matrix1.shape)
    confusion_matrix2=confusion_matrix(Y_Df_test,Y_Df_pred)

    print(confusion_matrix2.shape)
    print(pd.crosstab(Y_Df_test, Y_Df_pred, rownames=['Actual attacks'], colnames=['Predicted attacks']))
    


    accuracy = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
    precision = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='precision')
    print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
    recall = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='recall')
    print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
    f = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='f1')
    print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
    print("train_time:%.3fs\n" %train1)
    print("test_time:%.3fs\n" %test1)


    # Save trained model
    with open('IDS_model_DECISION TREE CLASSIFIER.pkl', 'wb') as file:
        pickle.dump(clf_Tree, file)

    ''' #SUPPORT VECTOR MACHINE
    clf_svm= svm.SVC()
    train_svm0=time.time()
    #traning DT
    clf_svm=clf_svm.fit(X_Df,Y_Df.astype(int))
    train_svm1=time.time()-train_svm0

    test_svm0=time.time()
    Y_Df_pred_svm=clf_svm.predict(X_Df_test)
    test_svm1 = time.time() - test_svm0


    accuracy = cross_val_score(clf_svm, X_Df_test, Y_Df_test, cv=10, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
    precision = cross_val_score(clf_svm, X_Df_test, Y_Df_test, cv=10, scoring='precision')
    print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
    recall = cross_val_score(clf_svm, X_Df_test, Y_Df_test, cv=10, scoring='recall')
    print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
    f = cross_val_score(clf_svm, X_Df_test, Y_Df_test, cv=10, scoring='f1')
    print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
    print("train_time:%.3fs\n" %train_svm1)
    print("test_time:%.3fs\n" %test_svm1)
    
    # Create confusion matrix
    pd.crosstab(Y_Df_test, Y_Df_pred_svm, rownames=['Actual attacks'], colnames=['Predicted attacks'])

    print('confusion matrix for SVM')
    print(pd.crosstab(Y_Df_test, Y_Df_pred_svm, rownames=['Actual attacks'], colnames=['Predicted attacks']))


    # Save trained model
    with open('IDS_model_SUPPORT VECTOR MACHINE.pkl', 'wb') as file:
        pickle.dump(clf_svm, file)

    '''


    

    return df

print(load_dataset())