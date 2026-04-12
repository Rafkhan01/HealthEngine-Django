import pickle
import os

# Get the directory where this predictor.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "disease_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
le_path = os.path.join(BASE_DIR, "label_encoder.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
le = pickle.load(open(le_path, "rb"))

def predict_disease(symptoms_list):
    text = " ".join(symptoms_list)
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    idx = probs.argmax()
    encoded = model.classes_[idx]
    disease = le.inverse_transform([encoded])[0]
    confidence = probs[idx]

    return disease, confidence