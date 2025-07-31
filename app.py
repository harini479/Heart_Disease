
#! C:/Users/harin/AppData/Local/Programs/Python/Python311/python.exe
print("Content-type:text/html\n\n");
from flask import Flask, request, render_template,url_for
import joblib
import pandas as pd
import webbrowser
import mysql.connector as my
def open():
       webbrowser.open('http://127.0.0.1:5000')

app1 = Flask(__name__)

@app1.route('/')
def registers():
    return render_template('index.html')

@app1.route('/sample2', methods=['GET', 'POST'])
def sample2():
       if request.method == 'POST':
              age = request.form['age']
              sex = request.form['sex']
              if sex == 'Male':
                     sex = 0
              else:
                     sex = 1
              cp = request.form['cp']
              if cp == '0':
                     cp = 0
              elif cp == '1':
                     cp = 1
              elif cp == '2':
                     cp = 2
              else:
                     cp = 3
              trestbps = request.form['trestbps']
              chol = request.form['chol']
              fbs = request.form['fbs']
              if fbs == '1':
                     fbs = 1
              else:
                     fbs = 0
              restecg = int(request.form['restecg'])
              thalach = request.form['thalach']
              exang = int(request.form['exang'])
              oldpeak = request.form['oldpeak']
              slope = request.form['slope']
              ca = request.form['ca']
              thal = int(request.form['thal'])

              # Ensure all values are float or int (Python native types)
              val = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), 
                     float(restecg), float(thalach), float(exang), float(oldpeak), 
                     float(slope), float(ca), float(thal)]

              # Load the machine learning model
              import numpy as np
              from sklearn.model_selection import train_test_split
              from sklearn.ensemble import RandomForestClassifier
              from sklearn.preprocessing import StandardScaler
              import pandas as pd

              # Reading and processing data
              data = pd.read_excel("heart.xls")
              X = data.drop(columns='target', axis=1)
              Y = data['target']
              X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

              sc = StandardScaler()
              X_train = sc.fit_transform(X_train)  # Normalize the features
              X_test = sc.transform(X_test)

              model = RandomForestClassifier()
              model.fit(X_train, Y_train)

              # Make prediction
              result = model.predict([val])
              # Output results
              if result[0] == 0:
                     resultss = 'No heart disease'
              else:
                     resultss = 'Heart disease detected'

              return render_template("/resultss.html", results=resultss)

       else:
              return render_template('index.html')


if __name__ == '__main__':
       open()
       app1.run(debug=False)
