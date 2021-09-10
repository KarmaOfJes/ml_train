import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns = 8
import pickle

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import sklearn
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    plot_roc_curve,
    plot_precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)
from sklearn.calibration import (
    calibration_curve
)
from scipy import stats
import random
import seaborn as sns


with open('Telco_churn\\RF\\fe_X_train.pickle', 'rb') as f:
    fe_X_train = pickle.load(f)
with open('Telco_churn\\RF\\fe_X_test.pickle', 'rb') as f:
    fe_X_test = pickle.load(f)
with open('Telco_churn\\RF\\y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)
with open('Telco_churn\\RF\\y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)
with open('Telco_churn\\RF\\fe_columns.pickle', 'rb') as f:
    fe_columns = pickle.load(f)


# Random Search
# set up the model
model = RandomForestClassifier(random_state=0)

# determine the hyper-parameter space
param_grid = dict(
    criterion=('gini', 'entropy'),
    n_estimators=stats.randint(50, 1000),
    max_depth=stats.randint(1, 100),
    min_samples_split=stats.uniform(0.01, 0.99),
)

# set up the search
search = RandomizedSearchCV(
    model,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_iter=100,
    random_state=10,
    n_jobs=-1,
    refit=True
)


search.fit(fe_X_train, y_train)
search.best_params_
train_predict = search.predict_proba(fe_X_train)[:, -1]
test_predict = search.predict_proba(fe_X_test)[:, -1]

print('Train set')
print('DT roc-auc: {}'.format(roc_auc_score(y_train, train_predict)))

print('Test set')
print('DTs roc-auc: {}'.format(roc_auc_score(y_test, test_predict)))

print(classification_report(y_test, search.predict(fe_X_test)))


# ROC Curve
plot_roc_curve(search, fe_X_test, y_test)


# Precision-recall curve
plot_precision_recall_curve(search, fe_X_test, y_test)


# Tuning a threshold
def tr_tuning(tr_array):
    df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'balanced_accuracy'])
    index = 0
    for threshold in tr_array:
        predictions = np.where(search.predict_proba(fe_X_test)[:, -1] < threshold, 0, 1)
        df = \
            pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            'threshold': threshold,
                            'precision': precision_score(y_test, predictions),
                            'recall': recall_score(y_test, predictions),
                            'f1': f1_score(y_test, predictions),
                            'balanced_accuracy': balanced_accuracy_score(y_test, predictions)
                        },
                        index=[index]
                    )
                ],
                axis=0
            )
        index += 1
    return df


tr_tuning(sorted(np.unique(search.predict_proba(fe_X_test)[:, -1])))


# Visualize parameter
cv_scores = search.cv_results_['mean_test_score']
cv_param = [_['max_depth'] for _ in [i for i in search.cv_results_['params']]]
sns.relplot(x=cv_param, y=cv_scores, kind='line')
plt.show()


# Show feature importance
pd.Series(search.best_estimator_.feature_importances_, index=fe_columns).sort_values(ascending=False)


# Calibration curve (return array of percentage of positive classes in each bin, and array of threshold bins)
calibration_curve(y_test, search.predict_proba(fe_X_test)[:, -1])





# Nested
X_train = pd.concat([fe_X_train, y_train], axis=1)
def nested_cross_val(model, grid):
    # configure the outer loop cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

    # configure the inner loop cross-validation procedure
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

    # enumerate splits
    outer_results = list()
    inner_results = list()

    for train_ix, test_ix in cv_outer.split(fe_X_train):
        # split data
        xtrain, xtest = fe_X_train.loc[train_ix, :], fe_X_train.loc[test_ix, :]
        ytrain, ytest = y_train[train_ix], y_train[test_ix]

        # define search
        search = GridSearchCV(
            model, grid, scoring='accuracy', cv=cv_inner, refit=True)

        # execute search
        search.fit(xtrain, ytrain)

        # evaluate model on the hold out dataset
        yhat = search.predict(xtest)

        # evaluate the model
        accuracy = f1_score(ytest, yhat)

        # store the result
        outer_results.append(accuracy)

        inner_results.append(search.best_score_)

        # report progress
        print(' >> accuracy_outer=%.3f, accuracy_inner=%.3f, cfg=%s' %
              (accuracy, search.best_score_, search.best_params_))

    # summarize the estimated performance of the model
    print()
    print('accuracy_outer: %.3f +- %.3f' %
          (np.mean(outer_results), np.std(outer_results)))
    print('accuracy_inner: %.3f +- %.3f' %
          (np.mean(inner_results), np.std(inner_results)))

    return search.fit(fe_X_train, y_train)

