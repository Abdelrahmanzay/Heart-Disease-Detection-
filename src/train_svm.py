from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(probability=True),
                        param_grid,
                        cv=5,
                        scoring='accuracy')

    grid.fit(X_train, y_train)
    print("Best SVM Params:", grid.best_params_)
    return grid.best_estimator_
