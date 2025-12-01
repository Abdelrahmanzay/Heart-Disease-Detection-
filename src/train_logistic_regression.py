from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=2000),
                        param_grid, cv=5, scoring='accuracy')

    grid.fit(X_train, y_train)
    print("Best Logistic Regression Params:", grid.best_params_)
    return grid.best_estimator_
