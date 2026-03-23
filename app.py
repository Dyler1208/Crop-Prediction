from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('Agui.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])

        sample = [[N, P, K, ph]]
        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)
        crop = le.inverse_transform(prediction)[0]

        return render_template('Agui.html', prediction_text=f"Recommended Crop: {crop}")

    except Exception as e:
        return render_template('Agui.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run()
