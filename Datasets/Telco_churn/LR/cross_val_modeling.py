import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns = 8
import pickle

from sklearn.linear_model import(
    LogisticRegression
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
import sklearn
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    GridSearchCV,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)
from scipy import stats
import random
import seaborn as sns


with open('Telco_churn\\LR\\fe_X_train.pickle', 'rb') as f:
    fe_X_train = pickle.load(f)
with open('Telco_churn\\LR\\fe_X_test.pickle', 'rb') as f:
    fe_X_test = pickle.load(f)
with open('Telco_churn\\LR\\y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)
with open('Telco_churn\\LR\\y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)
with open('Telco_churn\\LR\\fe_columns.pickle', 'rb') as f:
    fe_columns = pickle.load(f)


# Random Search
# set up the model
model = LogisticRegression(random_state=10, penalty='l2', solver='saga')

# determine the hyper-parameter space
param_grid = dict(
    C=stats.uniform(0, 50)
)

# cv = RepeatedKFold(
#     n_splits=5,
#     n_repeats=10,
#     random_state=10
# )
cv = KFold(
    n_splits=5,
    random_state=10,
    shuffle=True
)


# set up the search
search = RandomizedSearchCV(
    model,
    param_grid,
    scoring='roc_auc',
    cv=cv,
    n_iter=60,
    random_state=10,
    n_jobs=-1,
    refit=True
)


search.fit(fe_X_train, y_train)
search.best_params_
train_predict = search.predict_proba(fe_X_train)[:, -1]
test_predict = search.predict_proba(fe_X_test)[:, -1]


print('Train set')
print('GBM roc-auc: {}'.format(roc_auc_score(y_train, train_predict)))

print('Test set')
print('GBM roc-auc: {}'.format(roc_auc_score(y_test, test_predict)))

print(classification_report(y_test, search.predict(fe_X_test)))


# ROC Curve
y_true = y_test
y_proba = search.predict_proba(fe_X_test)[:, -1]
fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=0)
plt.plot(tpr, fpr)

# Print AUC
auc = np.trapz(fpr, tpr)
print('AUC:', auc)


# Visualize parameter
cv_scores = search.cv_results_['mean_test_score']
cv_param = [_['C'] for _ in [i for i in search.cv_results_['params']]]
sns.relplot(x=cv_param, y=cv_scores, kind='line')
plt.show()


# Show feature coefficients
pd.DataFrame(
    {
        'Coeff': pd.Series(search.best_estimator_.coef_[0], index=fe_columns),
        'Abs_coeff': np.abs(pd.Series(search.best_estimator_.coef_[0], index=fe_columns)),
        'Prob_coeff': pd.Series([np.exp(x)/(1 + np.exp(x)) for x in search.best_estimator_.coef_[0]], index=fe_columns)
    }
).sort_values('Prob_coeff', ascending=False)
# Intercept
search.best_estimator_.intercept_[0]


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


tuned_df = tr_tuning(sorted(np.unique(search.predict_proba(fe_X_test)[:, -1])))
tuned_df.iloc[
    np.arange(
        (tuned_df.loc[tuned_df.f1 == max(tuned_df.f1), :].index - 10)[0],
        (tuned_df.loc[tuned_df.f1 == max(tuned_df.f1), :].index + 11)[0]
    ),
    :
]





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
            model, grid, scoring='f1_score', cv=cv_inner, refit=True)

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

