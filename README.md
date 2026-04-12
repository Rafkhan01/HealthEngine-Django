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
[http://127.0.0.1:8000/ml_api/predict/?symptoms=1,0,1]
```

### Example Response

```json
{
  "prediction": "Urinary tract infection",
  "confidence": 0.033225840384561986
}
```

---

## Tech Stack

- Python 3.13
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
├── core/                          # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── ml_api/                        # Django app handling predictions
│   ├── views.py                   # Prediction logic and API response
│   ├── urls.py                    # Endpoint routing
│   ├── predictor.py               # ML model loading and inference
│   ├── apps.py
│   ├── models.py
│   ├── disease_model.pkl          # Trained Random Forest model
│   ├── label_encoder.pkl          # Fitted label encoder
│   ├── vectorizer.pkl             # Feature vectorizer
│   ├── dataset.csv                # Symptom-disease training dataset
│   ├── Symptom-severity.csv       # Symptom severity reference
│   ├── symptom_Description.csv    # Disease description reference
│   ├── symptom_precaution.csv     # Disease precaution reference
│   ├── medicine_filtered.csv      # Medicine recommendation dataset
│   └── requirements.txt
│
└── manage.py
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Rafkhan01/HealthEngine-Django.git
cd HealthEngine-Django
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r ml_api/requirements.txt
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

The trained model and supporting files are loaded at request time via `predictor.py` for clean separation of inference logic from the Django view layer.

---

## Related

- Original project repository (Streamlit version): https://github.com/Rafkhan01/AI-healthcare-system
- Live Streamlit demo: https://ai-healthcare-system-jaqgrqbayym7wsnbmvr4kf.streamlit.app/

---

## Author

Rafkhan B  
github.com/Rafkhan01
