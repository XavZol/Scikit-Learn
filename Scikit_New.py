from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

data = load_iris()
X = data.data
y = data.target

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.25, random_state=0)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', LogisticRegression())
])

pipeline.fit(X_entrena, y_entrena)

y_pred = pipeline.predict(X_prueba)
puntaje = pipeline.score(X_prueba, y_prueba)
print(f"Las predicciones son: {y_pred}")
print(f"La precisión del modelo es: {puntaje: .2f}")

