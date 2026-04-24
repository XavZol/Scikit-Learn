import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

data = np.array([[1 , -1, 2], 
                [2, 0, 0], 
                [0, 1, -1]])
print(data)

scaler = MinMaxScaler(feature_range=(0, 1))
data_escalada = scaler.fit_transform(data)
print(data_escalada)


iris = load_iris()
X = iris.data
X_escalado = scaler.fit_transform(X)

print(X[:5])
print(X_escalado[:5])

scaler2 = StandardScaler()
data_escalada2 = scaler2.fit_transform(data)
print(data_escalada2)

print(np.std(data_escalada2))

categorias = np.array([["rojo"], ["verde"], ["azul"], ["verde"], ["verde"], ["azul"]])
encoder = OneHotEncoder(sparse_output=False)
data_codificada = encoder.fit_transform(categorias)
print(data_codificada)

