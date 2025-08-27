from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train, y_train)
    models["LogisticRegression"] = logreg

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf
    
    return models