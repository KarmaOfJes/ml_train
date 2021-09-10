import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns = 8
import pickle

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)
import sklearn
from sklearn.model_selection import (
    KFold,
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
from sklearn.calibration import (
    calibration_curve
)
from scipy import stats
import random
import seaborn as sns


with open('Telco_churn\\DT\\fe_X_train.pickle', 'rb') as f:
    fe_X_train = pickle.load(f)
with open('Telco_churn\\DT\\fe_X_test.pickle', 'rb') as f:
    fe_X_test = pickle.load(f)
with open('Telco_churn\\DT\\y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)
with open('Telco_churn\\DT\\y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)
with open('Telco_churn\\DT\\fe_columns.pickle', 'rb') as f:
    fe_columns = pickle.load(f)


# Random Search
# set up the model
model = DecisionTreeClassifier(random_state=0)

# determine the hyper-parameter space
param_grid = dict(
    criterion=('gini', 'entropy'),
    max_depth=stats.randint(1, 40),
    min_samples_split=stats.uniform(0.01, 0.99)
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
y_true = y_test
y_proba = search.predict_proba(fe_X_test)[:, -1]
fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=0)
plt.plot(tpr, fpr)

# Print AUC
auc = np.trapz(fpr, tpr)
print('AUC:', auc)



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


# Decision Tree structure
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
search.best_estimator_.tree_.node_count
search.best_estimator_.tree_.feature
search.best_estimator_.tree_.threshold
search.best_estimator_.tree_.n_node_samples
search.best_estimator_.tree_.impurity
search.best_estimator_.tree_.children_left
search.best_estimator_.tree_.children_right

# print tree structure
n_nodes = search.best_estimator_.tree_.node_count
children_left = search.best_estimator_.tree_.children_left
children_right = search.best_estimator_.tree_.children_right
feature = search.best_estimator_.tree_.feature
threshold = search.best_estimator_.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has {n} nodes and has "
      "the following tree structure:\n".format(n=n_nodes))
for i in range(n_nodes):
    if is_leaves[i]:
        print("{space}node={node} is a leaf node.".format(
            space=node_depth[i] * "\t", node=i))
    else:
        print("{space}node={node} is a split node: "
              "go to node {left} if X[:, {feature}] <= {threshold} "
              "else to node {right}.".format(
                  space=node_depth[i] * "\t",
                  node=i,
                  left=children_left[i],
                  feature=feature[i],
                  threshold=threshold[i],
                  right=children_right[i]))


# Plot tree
plot_tree(search.best_estimator_)


# print decision path
node_indicator = search.best_estimator_.decision_path(fe_X_test)
leaf_id = search.best_estimator_.apply(fe_X_test)
sample_id = 0
# obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample {id}:\n'.format(id=sample_id))
for node_id in node_index:
    # continue to the next node if it is a leaf node
    if leaf_id[sample_id] == node_id:
        continue

    # check if value of the split feature for sample 0 is below threshold
    if (fe_X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
          "{inequality} {threshold})".format(
              node=node_id,
              sample=sample_id,
              feature=feature[node_id],
              value=fe_X_test.iloc[sample_id, feature[node_id]],
              inequality=threshold_sign,
              threshold=threshold[node_id]))






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

