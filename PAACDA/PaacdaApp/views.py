from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import math
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

global uname
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []
global dataset, ranges, classes, X, detected

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def Graphs(request):
    if request.method == 'GET':
        global graph_data
        output = "All Students Performance Graph"
        gd = np.asarray(graph_data)
        unique, count = np.unique(gd, return_counts=True)
        plt.pie(count,labels=unique,autopct='%1.1f%%')
        plt.title('Performance Graph')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def RunExtension(request):
    if request.method == 'GET':
        global dataset, ranges, y_test, classes
        y_test = dataset['Modified']
        X = dataset['Mystery_Data_Y'].ravel().astype(np.float64)
        X = X.reshape(-1, 1)
        X_train, X_test, y_train, y_tests = train_test_split(X, classes, test_size=0.2) #split data into train & test
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        calculateMetrics("Extension Hybrid PAACDA", pred, y_tests)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['LOF', 'Isolation Forest', 'One Class SVM', 'Propose PAACDA', 'Extension Hybrid PAACDA']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['LOF','Precision',precision[0]],['LOF','Recall',recall[0]],['LOF','F1 Score',fscore[0]],['LOF','Accuracy',accuracy[0]],
                           ['Isolation Forest','Precision',precision[1]],['Isolation Forest','Recall',recall[1]],['Isolation Forest','F1 Score',fscore[1]],['Isolation Forest','Accuracy',accuracy[1]],
                           ['One Class SVM','Precision',precision[2]],['One Class SVM','Recall',recall[2]],['One Class SVM','F1 Score',fscore[2]],['One Class SVM','Accuracy',accuracy[2]],
                           ['Propose PAACDA','Precision',precision[3]],['Propose PAACDA','Recall',recall[3]],['Propose PAACDA','F1 Score',fscore[3]],['Propose PAACDA','Accuracy',accuracy[3]],
                           ['Extension Hybrid PAACDA','Precision',precision[4]],['Extension Hybrid PAACDA','Recall',recall[4]],['Extension Hybrid PAACDA','F1 Score',fscore[4]],['Extension Hybrid PAACDA','Accuracy',accuracy[4]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)


def RunPaacda(request):
    if request.method == 'GET':
        global dataset, ranges, y_test, classes, X, detected
        y_test = dataset['Modified']
        corrupted = []
        detected = []
        values = dataset['Mystery_Data_Y'].values.ravel().astype(np.float64).tolist()
        print(values)
        print(type(values))
        start = 0
        for i in dataset['Mystery_Data_Y']:
            first=1
            second=1
            third=1
            for j in dataset['Mystery_Data_Y']:
                if(abs(j-i)<=ranges):
                    first=first+1
                if(abs(j-i)<=2*ranges):
                    second=second+1
                if(abs(j-i)<=3*ranges):
                    third=third+1
            index=0
            if(first!=1):
                index=(1/math.log(first))
            if(second!=1):
                index=index+(1/math.log(second))
            if(third!=1):
                index=index+(1/math.log(third))
            corrupted.append(index)
            if index > 0.70:
                detected.append(str(values[start])+" Corrupted")
            else:
                detected.append(str(values[start])+" Normal")
            start += 1    
        pred = []
        for value in corrupted:
            if value > 0.70:
                pred.append(True)
            else:
                pred.append(False)
        classes = pred
        for i in range(len(detected)):
            print(detected[i])
        calculateMetrics("Propose PAACDA", pred, y_test)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['LOF', 'Isolation Forest', 'One Class SVM', 'Propose PAACDA']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    

def RunOCS(request):
    if request.method == 'GET':
        global dataset, ranges, y_test
        y_test = dataset['Modified']
        X = dataset['Mystery_Data_Y'].ravel().astype(np.float64)
        X = X.reshape(-1, 1)
        ocs = OneClassSVM()
        ocs.fit(X)
        y_pred = ocs.predict(X)
        pred = []
        for value in y_pred:
            if value == 1:
                pred.append(False)
            else:
                pred.append(True)
        calculateMetrics("One Class SVM", pred, y_test)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['LOF', 'Isolation Forest', 'One Class SVM']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'ViewResult.html', context)      

def RunIsolation(request):
    if request.method == 'GET':
        global dataset, ranges, y_test
        y_test = dataset['Modified']
        X = dataset['Mystery_Data_Y'].ravel().astype(np.float64)
        X = X.reshape(-1, 1)
        iso = IsolationForest(n_estimators=1)
        iso.fit(X)
        y_pred = iso.predict(X)
        pred = []
        for value in y_pred:
            if value == 1:
                pred.append(False)
            else:
                pred.append(True)
        calculateMetrics("Isolation Forest", pred, y_test)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['LOF', 'Isolation Forest']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    

def RunLOF(request):
    if request.method == 'GET':
        global dataset, ranges, y_test
        y_test = dataset['Modified']
        X = dataset['Mystery_Data_Y'].ravel().astype(np.float64)
        X = X.reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
        lof.fit(X)
        y_pred = lof.predict(X)
        pred = []
        for value in y_pred:
            if value == 1:
                pred.append(False)
            else:
                pred.append(True)
        calculateMetrics("LOF", pred, y_test)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['LOF']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'ViewResult.html', context)        

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)          

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset, ranges
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("PaacdaApp/static/"+fname):
            os.remove("PaacdaApp/static/"+fname)
        with open("PaacdaApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("PaacdaApp/static/"+fname,nrows=1000)
        dataset.fillna(0, inplace = True)
        mean = dataset['Mystery_Data_Y'].mean(axis=0)
        ranges = mean/4
        columns = dataset.columns
        datasets = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(datasets)):
            output += '<tr>'
            for j in range(len(datasets[i])):
                output += '<td><font size="" color="black">'+str(datasets[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    







        
