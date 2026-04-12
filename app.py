import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("disease_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(
    page_title="AI Healthcare Decision Support System",
    page_icon="logo.png",
    layout="wide"
)

st.title(" AI Healthcare Decision Support System")
st.markdown("### Symptom-based Disease Prediction, Severity Analysis & Medicine Recommendation")

tabs = st.tabs([
    " Diagnosis",
    " Severity Analysis",
    " Treatment Recommendation",
    " Disease Info & Precautions"
])
data = pd.read_csv("dataset.csv")   
symptom_cols = [col for col in data.columns if col != "Disease"]
ss = pd.read_csv("Symptom-severity.csv")
severity_dict = dict(zip(ss['Symptom'].str.lower(), ss['weight']))
med_df = pd.read_csv("medicine_dataset.csv")
med_df['use_combined'] = med_df[['use0','use1','use2','use3','use4']].astype(str).agg(' '.join, axis=1).str.lower()
sd=pd.read_csv("symptom_Description.csv")
sp= pd.read_csv("symptom_precaution.csv")
# Collecting all unique symptoms
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(data[col].dropna().str.lower().str.strip())

all_symptoms_list = sorted(list(all_symptoms))

def predict_disease(symptoms_list):
    text = " ".join(symptoms_list)
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    idx = probs.argmax()
    encoded = model.classes_[idx]
    disease = le.inverse_transform([encoded])[0]
    confidence = probs[idx]

    return disease, confidence

def severity_level(score):
    if score <= 6:
        return "Low"
    elif score <= 10:
        return "Moderate"
    elif score <= 14:
        return "High"
    else:
        return "Critical"

def calculate_severity(symptoms):
    total = 0
    for s in symptoms:
        s = s.lower().replace(" ", "_")
        if s in severity_dict:
            total += severity_dict[s]
    return total, severity_level(total)

def recommend_medicine(disease):
    disease = disease.lower()
    matches = med_df[med_df['use_combined'].str.contains(disease, na=False)]

    if len(matches) == 0:
        return None

    matches['side_effect_count'] = matches.filter(like='sideEffect').notna().sum(axis=1)
    top = matches.sort_values('side_effect_count').head(3)

    return top[['name', 'Therapeutic Class', 'side_effect_count', 'substitute0', 'substitute1']]

with tabs[0]:
    st.subheader(" Disease Prediction")

    selected_symptoms = st.multiselect(
        " Search and select your symptoms",
        all_symptoms_list
    )

    if st.button("Analyze Health Condition"):
        if len(selected_symptoms) < 2:
            st.warning("Please select at least 2 symptoms.")
        else:
            st.session_state.selected_symptoms = selected_symptoms

            disease, confidence = predict_disease(selected_symptoms)
            st.session_state.predicted_disease = disease
            st.session_state.confidence = confidence

            st.success("Analysis Complete")
            st.metric("Predicted Disease", disease)
            st.progress(confidence)

with tabs[1]:
    st.subheader(" Symptom Severity Analysis")

    if 'selected_symptoms' in st.session_state and len(st.session_state.selected_symptoms) > 1:
        score, level = calculate_severity(st.session_state.selected_symptoms)

        st.metric("Risk Score", score)
        st.metric("Severity Level", level)

        if level == "Low":
            st.success("Low risk. Monitor symptoms.")
        elif level == "Moderate":
            st.warning("Moderate risk. Medical consultation advised.")
        elif level == "High":
            st.error("High risk. Visit hospital soon.")
        else:
            st.error("CRITICAL CONDITION! Immediate emergency care required.")

with tabs[2]:
    st.subheader(" Treatment Recommendation")

    if 'predicted_disease' in st.session_state:
        meds = recommend_medicine(st.session_state.predicted_disease)

        if meds is not None:
            st.dataframe(meds)
            st.warning(" Educational purpose only. Consult a doctor.")
        else:
            st.info("No exact medicine found in dataset.")

with tabs[3]:
    st.subheader("Disease Description & Precautions")
    if 'predicted_disease' in st.session_state:
        disease = st.session_state.predicted_disease
        desc = sd[sd['Disease'] == disease]['Description'].values

        if len(desc) > 0:
            st.write(desc[0])
        else:
            st.warning("Description not found.")
    if 'predicted_disease' in st.session_state:
        disease = st.session_state.predicted_disease
        row = sp[sp['Disease'] == disease]

        if not row.empty:
            for i in range(1, 5):
                st.write("•", row[f'Precaution_{i}'].values[0])
        else:
            st.warning("Precaution data not found.")
