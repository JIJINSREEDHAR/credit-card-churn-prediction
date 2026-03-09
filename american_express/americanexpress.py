#liberies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import chardet
import joblib




# Load dataset
with open("AmericanExpress.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # detect encoding on first 100KB
    print(result)
data_set = pd.read_csv("AmericanExpress.csv", encoding=result["encoding"])
x = data_set.iloc[:, 0:-1].values
y = data_set.iloc[:, -1].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # use transform, not fit_transform on test
# save encoders and scaler
joblib.dump(le, "gender_encoder.pkl")
joblib.dump(ct, "geography_encoder.pkl")
joblib.dump(sc, "scaler.pkl")

# Building the ANN
ann = tf.keras.models.Sequential()
from tensorflow.keras.layers import Dropout, BatchNormalization

# adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3)) 
# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3)) 
# adding the output layer (binary classification)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ann training
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, validation_split=0.2, #callbacks=[
   # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)]
   )



# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)  # threshold for binary classification

# printing predictions vs actual
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc)

# After training your ANN
ann.save("ann_model.keras")

