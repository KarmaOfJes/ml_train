import pandas as pd
pd.options.display.max_columns = 8
import pickle

# for the model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# for feature engineering
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
from feature_engine import transformation as vt
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as sp


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

# Form pipeline
fe_pipeline = Pipeline([
    (
        'imputer_num',
        mdi.AddMissingIndicator(variables=na_num_cols)
    ),
    (
        'cat_encoder',
        ce.OrdinalEncoder(variables=cat_cols)
    )
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

# Transform train/test
fe_X_train = fe_pipeline.transform(X_train)
fe_X_test = fe_pipeline.transform(X_test)

fe_columns = fe_X_train.columns

with open('Telco_churn\\RF\\fe_X_train.pickle', 'wb') as f:
    pickle.dump(fe_X_train, f)
with open('Telco_churn\\RF\\fe_X_test.pickle', 'wb') as f:
    pickle.dump(fe_X_test, f)
with open('Telco_churn\\RF\\y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
with open('Telco_churn\\RF\\y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)
with open('Telco_churn\\RF\\fe_columns.pickle', 'wb') as f:
    pickle.dump(fe_columns, f)
