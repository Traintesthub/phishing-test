import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import time

app = Flask(__name__)
model = pickle.load(open('xgbClass.pkl', 'rb'))
#model_log = pickle.load(open('logmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
    
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = model.predict(final_feature)
    
    if prediction == 0:
        output = 'Not a Phishing webpage'
    elif prediction == 1:
        output = 'Alert!!! Phishing Webpage Detected'
    return render_template('index.html', prediction_text = output)


@app.route('/predict_log', methods=['POST'])
def predict_log():
    int_features = [float(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = model_log.predict(final_feature)
    if prediction == 1:
        output = 'good'
    elif prediction == 0:
        output = 'poorly'
    return render_template('index.html', prediction_text = 'student is performing: {}'.format(output))


@app.route('/predict_api', methods=['GET'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
