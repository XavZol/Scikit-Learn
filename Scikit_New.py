from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X = data.data
y = data.target

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamaño del conjunto total:", len(X))
print("Tamaño del conjunto entrenamiento: ", len(X_entrena))
print("Tamaño del conjunto pruebas: ", len(X_prueba))

selector = SelectKBest(chi2, k=2)
X_nuevo = selector.fit_transform(X_entrena, y_entrena)
print(X_entrena[:5])
print(X_nuevo[:5])

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
selector2 = SelectFromModel(modelo)
X_importante = selector2.fit_transform(X_entrena, y_entrena)
print(X_entrena[:5])
print(X_importante[:5])