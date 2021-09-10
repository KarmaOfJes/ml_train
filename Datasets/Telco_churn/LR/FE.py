import pandas as pd
pd.options.display.max_columns = 8
import pickle
# for Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

# for feature engineering
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
import feature_engine.transformation as vt
import sklearn.preprocessing as sp
from sklearn.preprocessing import StandardScaler


# Load pickled files
with open('Telco_churn\\dataset_da.pickle', 'rb') as f:
    df = pickle.load(f)
with open('Telco_churn\\cat_cols.pickle', 'rb') as f:
    cat_cols = pickle.load(f)
with open('Telco_churn\\num_cols.pickle', 'rb') as f:
    num_cols = pickle.load(f)
with open('Telco_churn\\na_cat_cols.pickle', 'rb') as f:
    na_cat_cols = pickle.load(f)
with open('Telco_churn\\na_num_cols.pickle', 'rb') as f:
    na_num_cols = pickle.load(f)
with open('Telco_churn\\cat_otl_cols.pickle', 'rb') as f:
    cat_rare_cols = pickle.load(f)
with open('Telco_churn\\num_otl_cols.pickle', 'rb') as f:
    num_otl_cols = pickle.load(f)


# Reset index to customerID
df.set_index('customerID', inplace=True)
df.info()


# Create custom sklearn transformer to encode DecisionTreeDiscretiser output into 'object' type
fe_columns = None  # To populate with featurenames
class CustomCateg(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def change_type(self, x):
        x = x.astype('object', copy=True)
        return x

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        global fe_columns
        fe_columns = X.columns
        X_c = X.copy()
        for _ in self.columns:
            X_c[_] = self.change_type(X_c[_])
        return X_c



# Form pipeline
fe_pipeline = Pipeline([
    (
        'imputer_num',
        mdi.MeanMedianImputer(imputation_method='median', variables=na_num_cols)
    ),
    (
        'categorical_encoder',
        ce.OneHotEncoder(variables=cat_cols)
    ),
    (
        'numerical_discretizer',
        dsc.DecisionTreeDiscretiser(
            variables=num_cols,
            cv=10,
            regression=False,
            scoring='roc_auc',
            param_grid={
                'max_depth': [1,2,3,4,5,6,7,8,9,10]
            }
        )
    ),
    (
        'bin_categ',
        CustomCateg(num_cols)
    )#,
    # (
    #     'final_enc',
    #     ce.OrdinalEncoder(variables=num_cols)
    # )
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Churn', axis=1),
    df['Churn'],
    test_size=0.2,
    random_state=0
)
X_train.shape, X_test.shape


# Fit FE pipeline
fe_pipeline.fit(X_train, y_train)
fe_pipeline.named_steps['numerical_discretizer'].binner_dict_['TotalCharges'].best_params_


# Transform train/test
fe_X_train = fe_pipeline.transform(X_train)
fe_X_test = fe_pipeline.transform(X_test)


# Use sklearn StandarScaler lastly, to apply column names saved previously after feature-engine pipeline
sc = StandardScaler().fit(fe_X_train)
fe_X_train = sc.transform(fe_X_train)
fe_X_test = sc.transform(fe_X_test)


with open('Telco_churn\\LR\\fe_X_train.pickle', 'wb') as f:
    pickle.dump(fe_X_train, f)
with open('Telco_churn\\LR\\fe_X_test.pickle', 'wb') as f:
    pickle.dump(fe_X_test, f)
with open('Telco_churn\\LR\\y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
with open('Telco_churn\\LR\\y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)
with open('Telco_churn\\LR\\fe_columns.pickle', 'wb') as f:
    pickle.dump(fe_columns, f)
