from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "⚠️ Patient is at risk of death" if prediction == 1 else "✅ Patient is not at risk"
        return render_template('index3.html', prediction_text=result)
    except Exception as e:
        return render_template('index3.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)

