# train.py
# Trains a simple RandomForest classifier on the classic Iris dataset
# and saves it to disk so the serving container never has to retrain.

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the built-in Iris dataset (150 samples, 4 features, 3 classes)
X, y = load_iris(return_X_y=True)

# Fit a RandomForest classifier. random_state is fixed for reproducibility.
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# Persist the trained model so app.py can load it without retraining.
joblib.dump(model, 'model.joblib')
print('Saved model.joblib')
