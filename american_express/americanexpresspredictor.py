#liberies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import chardet
import joblib
#load ANN model
ann = tf.keras.models.load_model("ann_model.keras")

#user input for prediction
print("Enter the values for the features:")
feature1 = float(input("Enter value for credit score(numeric): "))
feature2 = input("Enter value for Geography(categorical): ").strip().capitalize()
feature3 = input("Enter value for Gender (categorical): ").strip().capitalize()
feature4 = float(input("Enter value for Age (numeric): "))
feature5 = float(input("Enter value for customer since (numeric): "))
feature6 = float(input("Enter value for current account (numeric): "))
feature7 = float(input("Enter value for num of prodects (numeric): "))
feature8 = float(input("Enter value for upi enabled (numeric 1=yes,0=no): "))
feature9 = float(input("Enter value for annual income (numeric): "))

# Prepare input data for prediction
input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]])

# Encoding categorical data
le = joblib.load("gender_encoder.pkl")
ct = joblib.load("geography_encoder.pkl")
sc = joblib.load("scaler.pkl")

input_data[0, 2] = le.transform([input_data[0, 2]])[0]  


input_data= np.array(ct.transform(input_data))

#feature scaling

input_data = sc.transform(input_data)

# Predicting the result
prediction = ann.predict(input_data)
prediction = (prediction > 0.5)  
# Displaying the prediction result
if prediction[0][0]:
    print("The customer is likely to close the credit card.") 
else:
    print("The customer is likely to keep the credit card.")


