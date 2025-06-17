from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os
from django.conf import settings

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    try:
        # Load dataset (use a relative path for production)
        data_path = os.path.join(settings.BASE_DIR, 'data', 'diabetes.csv')
        df = pd.read_csv(r"C:\Users\Dell\Desktop\mlt_proj\dataset\diabetes.csv")

        
        # Prepare data
        X = df.drop("Outcome", axis=1)
        Y = df['Outcome']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize and train Gradient Boosting Classifier
        gbm = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gbm.fit(X_train, Y_train)
        
        # Get user input from the form
        val1 = float(request.GET.get('n1', 0))
        val2 = float(request.GET.get('n2', 0))
        val3 = float(request.GET.get('n3', 0))
        val4 = float(request.GET.get('n4', 0))
        val5 = float(request.GET.get('n5', 0))
        val6 = float(request.GET.get('n6', 0))
        val7 = float(request.GET.get('n7', 0))
        val8 = float(request.GET.get('n8', 0))
        
        # Scale the input features
        input_data = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])
        
        # Make prediction
        pred = gbm.predict(input_data)
        proba = gbm.predict_proba(input_data)[0]  # Get probability scores
        
        # Prepare result
        result_status = 'positive' if pred[0] == 1 else 'negative'
        confidence = proba[1] if pred[0] == 1 else proba[0]
        
        return render(request, "predict.html", {
            "result2": result_status,
            "confidence": f"{confidence*100:.2f}%",
            "model_name": "Gradient Boosting Machine"
        })
        
    except Exception as e:
        return render(request, "predict.html", {
            "error": f"An error occurred: {str(e)}"
        })