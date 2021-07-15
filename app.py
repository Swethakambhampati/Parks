from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import jsonify

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('forest_fire.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = StandardScaler.transform(final_features)    
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)

    if output == 0:
        return render_template('forest_fire.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE PARKINSONS')
    else:
         return render_template('forest_fire.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A PARKINSONS')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)





if __name__ == '__main__':
    app.run(debug=False)