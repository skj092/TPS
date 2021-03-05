from sklearn import linear_model
from sklearn import ensemble

models = {
    "logistic_regression": linear_model.LogisticRegression(),
    "random_forest": ensemble.RandomForestClassifier(n_jobs=-1),
}
