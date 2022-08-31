
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import joblib



df = pd.read_csv("data.csv")

X = df[["Flour", "Milk","Sugar","Butter","Egg","Baking Powder","Vanilla","Salt"]]
Y = df["Type"]

model = GradientBoostingClassifier() 
model.fit(X, Y)
joblib.dump(model,"clf.pkl")