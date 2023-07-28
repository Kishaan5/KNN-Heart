from flask import Flask, render_template, request
import pickle
import numpy as np


model1 = pickle.load(open(r"C:\Users\Kishaan\Documents\projects\Heart Disease KNN\Model1.pkl", "rb"))

app1 = Flask(__name__)


@app1.route("/")
def man():
    return render_template("home.html")



@app1.route("/predict",methods=["POST"])
def home():
    data1 = request.form["a"] 
    data2 = request.form["b"] 
    data3 = request.form["c"] 
    data4 = request.form["d"] 
    data5 = request.form["e"] 
    data6 = request.form["f"] 
    data7 = request.form["g"]  
    arr = np.array([[data1 ,data2 ,data3 ,data4 , data5 , data6, data7, ]])
    pred = model1.predict(arr)
    return render_template("after.html", data=pred)

if __name__ == "__main__":
    app1.run(debug=True)