# Disease Prediction API

A Django-based REST API that wraps a trained machine learning model to predict diseases from binary symptom inputs. This project extends the original [AI Healthcare System](https://github.com/Rafkhan01/AI-healthcare-system) by exposing the ML pipeline as a queryable HTTP endpoint.

Live demo of the original Streamlit application: https://ai-healthcare-system-jaqgrqbayym7wsnbmvr4kf.streamlit.app/

---

## Overview

The core ML model was trained using Scikit-learn on a labeled symptom-disease dataset. This repository wraps that model in a Django application, making predictions accessible via a simple GET request with symptom flags passed as query parameters.

---

## API Endpoint

```
GET /ml_api/predict/?symptoms=<binary_symptom_vector>
```

### Parameters

| Parameter | Type   | Description                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| symptoms  | string | Comma-separated binary values (1 or 0) representing presence or absence of each symptom in the dataset order |

### Example Request

```
http://127.0.0.1:8000/ml_api/predict/?symptoms=1,0,1,0,0,1
```

### Example Response

```json
{
  "predicted_disease": "Fungal infection",
  "confidence": 0.94
}
```

---

## Tech Stack

- Python 3.x
- Django
- Scikit-learn
- Pandas
- NumPy
- Pickle (model serialization)

---

## Project Structure

```
disease-prediction-django-api/
│
├── disease_api/               # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── ml_api/                    # Django app handling predictions
│   ├── views.py               # Prediction logic and API response
│   ├── urls.py                # Endpoint routing
│   └── apps.py
│
├── models/                    # Serialized ML model files
│   ├── disease_model.pkl
│   └── label_encoder.pkl
│
├── manage.py
└── requirements.txt
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Rafkhan01/disease-prediction-django-api.git
cd disease-prediction-django-api
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the development server

```bash
python manage.py runserver
```

### 4. Query the API

Open a browser or use curl:

```bash
curl "http://127.0.0.1:8000/ml_api/predict/?symptoms=1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
```

---

## ML Model Details

- Algorithm: Random Forest Classifier (Scikit-learn)
- Dataset: Symptom-disease labeled dataset with 132 symptom features
- Accuracy: 85%+ classification accuracy on symptom-disease mapping
- Output: Predicted disease label decoded via a fitted LabelEncoder

The trained model (`disease_model.pkl`) and encoder (`label_encoder.pkl`) are loaded once at application startup for efficient repeated inference.

---

## Related

- Original project repository (Streamlit version): https://github.com/Rafkhan01/AI-healthcare-system
- Live Streamlit demo: https://ai-healthcare-system-jaqgrqbayym7wsnbmvr4kf.streamlit.app/

---

## Author

Rafkhan B  
github.com/Rafkhan01
