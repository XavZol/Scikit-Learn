from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

data = load_iris()
X = data.data
y = data.target

modelo = RandomForestClassifier(n_estimators=100, random_state=42)

puntaje = cross_val_score(modelo, 
                            X, 
                            y,
                            cv=5)

print("Exactitud de cada partición: ", puntaje)
print("Media de la exactitud:", puntaje.mean())