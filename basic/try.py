import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chardet

with open("covid.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # detect encoding on first 100KB
    print(result)
data_set = pd.read_csv("covid.csv", encoding=result["encoding"])
x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, -1].values

#print(x)
#print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(x[:, 0:1])
x[: , 0:1] = imputer.transform(x[:, 0:1])

imputer.fit(x[:, 4:5])
x[: , 4:5] = imputer.transform(x[:, 4:5])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [1])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

#print(x)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 6:] = sc.fit_transform(x_train[:, 6:])
x_test[:, 6:] = sc.fit_transform(x_test[:, 6:])

print(x_train)
print(x_test)