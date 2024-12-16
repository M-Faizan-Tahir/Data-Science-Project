import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

st.markdown("""
    <style>
        .centered-message {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background-color: #085820;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .centered-message-red {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background-color: #880015;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container {
            max-width: 1200px;  /* Increase this value to make the page wider */
            padding-left: 5%;   /* Optional: Adjust padding to center the content */
            padding-right: 5%;  /* Optional: Adjust padding */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style="font-size: 50px;">Diabetes Prediction Model <sub style="color: grey; font-size: 16px;">by Muhammad Faizan Tahir [DATA SCIENTIST]</sub></h2>
""", unsafe_allow_html=True)

dataset_path = "diabetes.csv"

# Load dataset if the path is provided
if dataset_path:
    try:
        data = pd.read_csv(dataset_path)
        st.write("Dataset Preview:")
        st.write(data.head())

        # Prepare features and target variable
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Encode the target labels
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        # Handle missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # User input for prediction
        

        
        with st.expander("Enter your details to predict if you have diabetes:"):
            pregnancies = st.slider("Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.slider("Glucose Level", min_value=0, max_value=300, value=100)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            insulin = st.slider("Insulin Level", min_value=0, max_value=800, value=80)
            bmi = st.slider("BMI", min_value=0.0, max_value=100.0, value=22.0)
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
            age = st.slider("Age", min_value=18, max_value=100, value=30)

    
    
        # Default healthy values for sliders (based on general healthy values)
        

        # Create the input data array
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

        # Impute missing values for user input
        user_input = imputer.transform(user_input)

        # Predict probability of not having diabetes
        Prediction = knn.predict(user_input)
        probability = knn.predict_proba(user_input)

        
        prob_diabetes = probability[0][1]
        probability_DB = prob_diabetes * 100

        # Display final message based on the probability
        if Prediction == 1:
            st.markdown(f'<div class="centered-message-red">Based on the data provided, you are diabetic</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="centered-message">Based on the data provided, you are not diabetic, but your probability of becoming diabetic is {probability_DB}%</div>', unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error loading the dataset: {e}")
else:
    st.write("Please enter the path to your dataset.")
