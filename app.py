import base64
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, flash, render_template,request,redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=1)

app.config['SECRET_KEY']='supersecret'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.debug = True

route_accessed = {"upload_KNN": False, "upload_DecisionTree": False, "upload_NaiveBayes": False}

#for generating the foilders for the confusion matrices
confusion_matrix_folder = 'Confusion Matrices'
confusion_matrix_decisionTree = 'Confusion Matrices/Confusion Matrices Decision Tree'
confusion_matrix_KNN = 'Confusion Matrices/Confusion Matrices KNN'
confusion_matrix_NaiveBayes = 'Confusion Matrices/confusion Matrices NaiveBayes'

if not os.path.exists(confusion_matrix_folder):
    os.makedirs(confusion_matrix_folder)

confusion_matrices_directories = [confusion_matrix_decisionTree,confusion_matrix_KNN,confusion_matrix_NaiveBayes]
for x in confusion_matrices_directories:
    if not os.path.exists(x):
        os.makedirs(x)


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload FIle")


@app.route("/")
def home():
    return render_template("index.html", model_url = url_for)

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/upload_KNN")
def upload_KNN():
    route_accessed["upload_KNN"] = True
    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_DecisionTree")
def upload_DecisionTree():
    route_accessed["upload_DecisionTree"]=True
    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_NaiveBayes")
def upload_NaiveBayes():
    route_accessed["upload_NaiveBayes"] = True
    form = UploadFileForm()
    return render_template('upload.html', form=form)


@app.route('/results', methods=['POST'])
def submit():
    from intrusion_detection import load_dataset, col_names,confusion_matrix,categorical_columns

    file = request.files['file']
    file_path = "/" + file.filename
    file.save("dataset.csv")
    load_dataset('dataset.csv')

    executor.submit(process,file)
   
    if request.method == 'POST':
        #check for if file is empty
        if file.filename == '':
            return "Error: No file selected for upload"
        
        #read the uploaded dataset
        try:
            df = pd.read_csv('dataset.csv', header=None, names=col_names)
        except pd.errors.EmptyDataError:
            return "Error: Uploaded file is empty or contains no data"
        
        #check is dataframe is empty
        if df.empty:
            return "Error: Uploaded file is empty"
        
    return render_template('result.html')

def process():
 # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    if route_accessed["upload_DecisionTree"] == True:
            from intrusion_detection import DecisionTree
            print(route_accessed)
            route_accessed["upload_DecisionTree"]=False
            print(route_accessed)
            
            
            #load the trained model
            with open('IDS_model_DECISION TREE CLASSIFIER.pkl', "rb") as file:
                clf = pickle.load(file)

            confusion_matrixDecisionTreeClassifier,metrics = DecisionTree()

            #performance metrics
            accuracy = metrics['Accuracy']
            Accuracy ="ACCURACY: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2)

            precision = metrics['Precision']
            Precision = "PRECISION: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2)

            recall = metrics['Recall']
            Recall = "RECALL: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2)

            f = metrics['F-measure']
            Fm = "F-MEASURE: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2)

            Train = metrics['Train']
            Train_time = "TRAIN TIME:%.3fs\n" %Train

            test = metrics['Test']
            Test_time = "TEST TIME:%.3fs\n" %test


            # Save confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrixDecisionTreeClassifier, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Attacks')
            plt.ylabel('Actual Attacks')
            results = "CONFUSION MATRIX FOR THE DECISION TREE MODEL"
            #plt.tight_layout()

            #define the new file name with the timestamp
            filename = f'confusion_matrixDecisionTree({timestamp}).png'
            plt.savefig(filename)

            # Convert plot to base64 for display in HTML
            with open(filename, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            if os.path.exists(filename):
                shutil.move(filename, os.path.join(confusion_matrix_decisionTree, filename))

            os.remove('dataset.csv')  # Remove uploaded file
            # Return data as JSON
            return jsonify({'confusion_matrix': img_base64, 'results': results, 'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'Fm':Fm,'Train':Train_time, 'Test':Test_time})
            # return render_template('result.html', confusion_matrix=img_base64, results=results)

    elif route_accessed["upload_KNN"] == True:
            from intrusion_detection import KNN
            print(route_accessed)
            route_accessed["upload_KNN"]=False
            print(route_accessed)
            #load the trained model
            with open('IDS_model_KNN.pkl', "rb") as file:
                clf = pickle.load(file)

                confusion_matrixKNN,metrics = KNN()
                 #performance metrics
                accuracy = metrics['Accuracy']
                Accuracy ="ACCURACY: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2)

                precision = metrics['Precision']
                Precision = "PRECISION: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2)

                recall = metrics['Recall']
                Recall = "RECALL: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2)

                f = metrics['F-measure']
                Fm = "F-MEASURE: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2)

                Train = metrics['Train']
                Train_time = "TRAIN TIME:%.3fs\n" %Train

                test = metrics['Test']
                Test_time = "TEST TIME:%.3fs\n" %test

                 # Save confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrixKNN, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Attacks')
                plt.ylabel('Actual Attacks')
                results = "CONFUSION MATRIX FOR THE KNN MODEL"
                 #define the new file name with the timestamp
                filename = f'confusion_matrixKNN({timestamp}).png'
                plt.savefig(filename)
                #plt.tight_layout()
                plt.savefig(filename)

                # Convert plot to base64 for display in HTML
                with open(filename, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                if os.path.exists(filename):
                    shutil.move(filename, os.path.join(confusion_matrix_KNN, filename))


                # Return data as JSON
                return jsonify({'confusion_matrix': img_base64, 'results': results, 'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'Fm':Fm,'Train':Train_time, 'Test':Test_time})

    elif route_accessed["upload_NaiveBayes"] == True:
            from intrusion_detection import NaiveBayes
            print(route_accessed)
            route_accessed["upload_NaiveBayes"]=False
            print(route_accessed)

            confusion_matrixNaiveBayes,metrics = NaiveBayes()
             #performance metrics
            accuracy = metrics['Accuracy']
            Accuracy ="ACCURACY: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2)

            precision = metrics['Precision']
            Precision = "PRECISION: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2)

            recall = metrics['Recall']
            Recall = "RECALL: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2)

            f = metrics['F-measure']
            Fm = "F-MEASURE: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2)

            Train = metrics['Train']
            Train_time = "TRAIN TIME:%.3fs\n" %Train

            test = metrics['Test']
            Test_time = "TEST TIME:%.3fs\n" %test

            #load the trained model
            with open('IDS_model_NaiveBayes.pkl', "rb") as file:
                clf = pickle.load(file)

                
                #predictions = clf.predict(X_Df_Preprocessed)
                # Save confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrixNaiveBayes, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Attacks')
                plt.ylabel('Actual Attacks')
                #plt.tight_layout(
                
                 #define the new file name with the timestamp
                filename = f'confusion_matrixNaiveBayes({timestamp}).png'
                plt.savefig(filename)
                results = "CONFUSION MATRIX FOR THE NAIVE BAYES MODEL"

                # Convert plot to base64 for display in HTML
                with open(filename, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                if os.path.exists(filename):
                    shutil.move(filename, os.path.join(confusion_matrix_NaiveBayes, filename))

                # Return data as JSON
                return jsonify({'confusion_matrix': img_base64, 'results': results, 'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'Fm':Fm,'Train':Train_time, 'Test':Test_time})

    else:
        return render_template("model.html")
@app.route('/result')
def result():
    return process()

if __name__ == '__main__':
    app.run(port=6500)