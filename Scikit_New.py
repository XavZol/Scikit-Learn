from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

data = load_iris()
X = data.data
y = data.target

modelo = RandomForestClassifier(random_state=42)

parametros = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

mi_grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, scoring="accuracy")

mi_grid_search.fit(X, y)

print("Mejores Parametros:", mi_grid_search.best_params_)
print("Mejor exactitud:", mi_grid_search.best_score_)