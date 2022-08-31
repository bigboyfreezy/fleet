from flask import *
import joblib
import pandas

from flask import request
app.secret_key = 'A+4#s_T%P8g0@o?6'

import pymysql
app = Flask(__name__)



@app.route('/model', methods=['GET', 'POST'])
def model():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        Flour = request.form.get("Flour")
        Milk = request.form.get("Milk")
        Sugar = request.form.get("Sugar")
        Butter = request.form.get("Butter")
        Egg = request.form.get("Egg")
        BakingPowder= request.form.get("Baking Powder")
        Vanilla = request.form.get("Vanilla")
        Salt = request.form.get("Salt")

        
        # Put inputs to dataframe
        X = pandas.DataFrame([[Flour, Milk,Sugar,Butter,Egg,BakingPowder,Vanilla,Salt]], columns = ["Flour", "Milk","Sugar","Butter","Egg","Baking Powder","Vanilla","Salt"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("model.html", output = prediction)


@app.route('/model1', methods=['POST'])
def model1():
    from flask import request
    import numpy as np
    json = request.json
    spamm = json['spam']

    import pandas
    data = pandas.read_csv('spam.csv', encoding = "ISO-8859-1") 
    data = data[["v1","v2"]]
    from sklearn import model_selection
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data['v2'],data['v1'],test_size=0.3,random_state=42)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(lowercase=True,stop_words='english')
    X_new_train = vectorizer.fit_transform(X_train)
    from sklearn.linear_model import LogisticRegression,SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    

    model = SGDClassifier()
    model.fit(X_new_train,Y_train)
    new_data_vectorized = vectorizer.transform([spamm])
    outcome = model.predict(new_data_vectorized)
    out= str(outcome)
    
    
    response = jsonify(out)
    response.status_code = 200
    return response









if __name__ == '__main__':
    app.run(debug=True)