from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                        param_grid, cv=5, scoring='accuracy')

    grid.fit(X_train, y_train)
    print("Best Decision Tree Params:", grid.best_params_)
    return grid.best_estimator_
