# app.py
# Minimal Flask REST service that serves predictions from the
# pre-trained Iris model (model.joblib).

from flask import Flask, request, jsonify
import joblib, numpy as np

app = Flask(__name__)

# Load the trained model once at startup, not per-request, so
# predictions stay fast and the model is validated when the
# container starts (fail fast if model.joblib is missing/corrupt).
model = joblib.load('model.joblib')


@app.get('/')
def health():
    # Simple liveness/readiness check used by orchestrators
    # (Docker HEALTHCHECK, Kubernetes probes, load balancers, etc.)
    return {'status': 'ok'}


@app.post('/predict')
def predict():
    # Expects JSON body: {"features": [sepal_len, sepal_w, petal_len, petal_w]}
    data = request.get_json()
    x = np.array(data['features']).reshape(1, -1)
    pred = int(model.predict(x)[0])
    return jsonify({'prediction': pred})


if __name__ == '__main__':
    # host=0.0.0.0 is required so the Flask dev server accepts
    # connections from outside the container, not just localhost.
    app.run(host='0.0.0.0', port=5000)
