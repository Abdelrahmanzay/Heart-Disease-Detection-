from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid,
                        cv=5,
                        scoring='accuracy')

    grid.fit(X_train, y_train)
    print("Best KNN Params:", grid.best_params_)
    return grid.best_estimator_
