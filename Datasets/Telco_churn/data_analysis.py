import pandas as pd
import numpy as np
import re
pd.options.display.max_columns = 8
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import feature_engine.transformation as vt

# Loading data
df = pd.read_csv('Telco_churn\dataset.csv')
df
df.shape

# Target column is 'Churn', replace labels by integer values
df.Churn = np.where(df.Churn == 'Yes', 1, 0)
df.info()


# Create list with df columns excluding customerID and Churn
df_cols = df.columns.drop(['customerID', 'Churn'])

# Imbalanced dataset - 26.5% Churn
df.Churn.value_counts() / len(df.Churn)

# Detect number of distinct values per each column
col_types = {
    key: ('obj' if val < 20 else 'num')
    for key, val in ({
        var: df[var].nunique()
        for var in df_cols
    }).items()
}

# Recast types of variables
for var in col_types.items():
    if var[1] == 'obj':
        df[var[0]] = df[var[0]].astype('object', copy=False)
    elif var[1] == 'num':
        try:
            df[var[0]] = df[var[0]].astype('float64', copy=False)
        except ValueError:
            df[var[0]] = [i.replace(' ', '') for i in df[var[0]]]
            df[var[0]] = np.where(df[var[0]] == '', np.nan, df[var[0]])
            df[var[0]] = df[var[0]].astype('float64', copy=False)


"""Missing values"""
# Find Columns with NA
df.apply(lambda x: sum(x.isnull()), axis=0)
na_cols = [var for var in df_cols if df[var].isnull().sum() > 0]
na_cat_cols = [var for var in na_cols if bool(re.match('(object|category)', df[var].dtype.name))]
na_num_cols = [var for var in na_cols if bool(re.match('(int|float)', df[var].dtype.name))]

# Show proportion of Churn among missing values:
for var in na_cols:
    print(
        df.loc[df[var].isnull(), [var, 'Churn']].groupby('Churn').agg(
            CNT=('Churn', lambda x: x.count())
        ),
        '\n'
    )


# Visualize proportion:
for var in na_cols:
    temp_df = \
        pd.concat(
            [
                pd.DataFrame(
                    {
                        'Churn': df.loc[:, 'Churn'].unique(),
                        'CNT': [0, 0]
                     }
                ),
                df.loc[df[var].isnull(), [var, 'Churn']].groupby('Churn').agg(
                    CNT=('Churn', lambda x: x.count())
                ).reset_index()
            ]
        ).groupby('Churn')['CNT'].sum().reset_index()

    sns.catplot(x='Churn', y='CNT', hue='Churn', kind='bar', data=temp_df)
    plt.title(f'Proportion of Churn values among missing in {var}')
    plt.show()


# Dump NaN columns
with open('Telco_churn\\na_cat_cols.pickle', 'wb') as f:
    pickle.dump(na_cat_cols, f)
with open('Telco_churn\\na_num_cols.pickle', 'wb') as f:
    pickle.dump(na_num_cols, f)


# Find total percentage of na in column and proportion of missing among values of target
for var in na_cols:
    print(var)
    print('Percentage of missing: ', df[var].apply(lambda x: np.where(np.isnan(x), 1, 0)).mean())
    val_for_1 = df.loc[np.isnan(df[var]), 'Churn'].mean()
    print('Percentage of 1 target among missing: ', val_for_1)
    print('Percentage of 0 target among missing: ', 1 - val_for_1, '\n')


"""Numerical analysis:"""
num_cols = [var for var in df_cols if df[var].dtype.name not in ('object', 'category')]

# Dump numerical columns
with open('Telco_churn\\num_cols.pickle', 'wb') as f:
    pickle.dump(num_cols, f)

df[num_cols].apply(lambda x: x.nunique(), axis=0)
# tenure, MonthlyCharges, TotalCharges are continuous

# Explore relationship between target and numerical variables
df_num_expl = pd.DataFrame()
for var in num_cols:
    df_num_expl = \
        pd.concat(
            [
                df.groupby('Churn')[var].agg(
                    mean='mean',
                    median='median'
                ).stack().to_frame().reset_index().rename(
                    {
                        'level_1': 'metric',
                        0: 'val'
                    },
                    axis=1
                ).assign(
                    metric=lambda x: x['metric'] + f'_{var}'
                ),
                df_num_expl
            ]
        )
df_num_expl

# Bar-plot of mean/median for variable among target
for var in num_cols:
    temp_df = \
        df.groupby('Churn')[var].agg(
            mean='mean',
            median='median'
        ).stack().to_frame().reset_index().rename(
            {
                'level_1': 'metric',
                0: 'val'
            },
            axis=1
        )
    sns.catplot(x='Churn', y='val', hue='metric', kind='bar', data=temp_df)
    plt.title(f'Mean/Median of {var} per Churn')
    plt.show()

# Density plot of num variable among target
for var in num_cols:
    temp_df = \
        df.loc[:, [var, 'Churn']]
    # if col not empty it will divide into 2 plots (curve without filled color under it)
    # if multiple='stack'  it will stack two plots (curves with filled colors)
    # if both not empty plots would be filled with color (area under the curve)
    sns.displot(x=f'{var}', hue='Churn', col='Churn', multiple='stack', kind='kde', data=temp_df)
    plt.title(f'Density of {var} per Churn')
    plt.show()

# Inference:
# 1. SeniorCitizen 0: No Churn 76%, Yes Churn 23%
#    SeniorCitizen 1: No Churn 58%, Yes Churn 41%
# 2. Tenure: No Churn mean/median  = 37/38, Yes Churn 17/10
# 3. MonthlyCharges: No Churn mean/median  = 60/64, Yes Churn 74/79


"Categorical analysis"
df.dtypes
cat_cols = [var for var in df_cols if bool(re.match('(object|categ)', df[var].dtype.name))]


# Dump categorical columns
with open('Telco_churn\\cat_cols.pickle', 'wb') as f:
    pickle.dump(cat_cols, f)


# Count number for each unique value, and percentage
for var in cat_cols:
    print(
        pd.concat(
            [
                df[var].value_counts(),
                df[var].value_counts() / len(df[var])
            ],
            axis=1
        ),
        '\n'
    )


# Plot distribution of categories among Churn
for var in cat_cols:
    temp_df = df.groupby(['Churn', var])[var].agg(CNT='count').reset_index()
    sns.catplot(data=temp_df, x='Churn', y='CNT', kind='bar', hue=f'{var}')
    plt.title(f'Distribution of categories per variable {var}')
    plt.show()


# Show percentage of churn observations per each category value
for var in cat_cols:
    print(df.groupby(var)['Churn'].mean(), '\n')


"""Outliers"""
# Categorical:
# Show rare labels for category
cat_rare_cols = []
for var in cat_cols:
    print(var)
    otl = (df[var].value_counts() / len(df[var]))[lambda x: x < 0.02]  # !!!Special form of filtration!!!
    print(
        otl,
        '\n'
    )
    if otl.shape[0] > 0:
        cat_rare_cols.append(var)
# No rare categories

# Dump categorical columns with rare values
with open('Telco_churn\\cat_otl_cols.pickle', 'wb') as f:
    pickle.dump(cat_rare_cols, f)


# Numerical
# Show outliers
num_otl_cols = []
for var in num_cols:
    df = df.copy()
    df[var] = df.loc[:, [var]].fillna(df.loc[:, var].median())
    df[var] = vt.YeoJohnsonTransformer().fit_transform(df.loc[:, [var]])

    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1

    print(var)

    otl = \
        df[var][
            df[var].apply(
                lambda x:
                ((x < df[var].mean()) and (x < (Q1 - 1.5 * IQR)))
                or
                ((x > df[var].mean()) and (x > (Q3 + 1.5 * IQR)))
            )
        ].shape[0]

    print(otl, '\n')

    if otl > 0:
        num_otl_cols.append(var)
# No outliers
# Dump numerical columns with rare values
with open('Telco_churn\\num_otl_cols.pickle', 'wb') as f:
    pickle.dump(num_otl_cols, f)

# Export dataset
with open('Telco_churn\\dataset_da.pickle', 'wb') as f:
    pickle.dump(df, f)
