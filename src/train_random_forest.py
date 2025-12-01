from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15]
    }

    grid = GridSearchCV(RandomForestClassifier(),
                        param_grid,
                        cv=5,
                        scoring='accuracy')

    grid.fit(X_train, y_train)
    print("Best RF Params:", grid.best_params_)
    return grid.best_estimator_
