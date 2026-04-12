from django.shortcuts import render
from django.http import JsonResponse
from .predictor import predict_disease

def predict(request):
    symptoms = request.GET.get('symptoms')
    
    if not symptoms:
        return JsonResponse({"error": "No symptoms provided"}, status=400)
        
    symptoms_list = [x.strip() for x in symptoms.split(',')]
    
    disease, confidence = predict_disease(symptoms_list)
    
    return JsonResponse({
        "prediction": disease,
        "confidence": float(confidence)
    })
