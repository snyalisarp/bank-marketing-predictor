import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# data source: https://archive.ics.uci.edu/dataset/222/bank+marketing
full_data = pd.read_csv("bank-full.csv", sep=';')
sample_data = pd.read_csv("bank.csv", sep=';')
df_concat = pd.merge(full_data, sample_data, how='left', indicator=True)
train_data = df_concat[df_concat['_merge'] == 'left_only'].drop('_merge', axis=1)

train_df = train_data.copy()
test_df = sample_data.copy()
train_df.drop('duration', axis=1,inplace=True)
test_df.drop('duration', axis=1,inplace=True)

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_temp = X.copy()
        X_temp = self._do_transform(X_temp)
        self.feature_names_out_ = X_temp.columns.tolist()
        return self
    
    def transform(self, X):
        return self._do_transform(X)

    def _do_transform(self, X):
        X = X.copy()
        yes_no_map = {'no': 0, 'yes': 1}
        
        
        for col in ['default', 'housing', 'loan']:
            X[col] = X[col].map(yes_no_map)
            
        
        X['is_non_negative_balance'] = (X['balance'] >= 0).astype(int)
        X['new_client'] = (X['pdays'] == -1).astype(int)
        
        
        month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                     'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        
        m_idx = X['month'].map(month_map)
        X['month_sin'] = np.sin(2 * np.pi * m_idx / 12)
        X['month_cos'] = np.cos(2 * np.pi * m_idx / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        
        
        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
        for col in cat_cols:
            X[col] = X[col].astype('category')
            
        return X.drop(['month', 'day'], axis=1)

    # necessary method for set_output
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


preprocessor = ColumnTransformer(
    transformers=[
        ('num_scaling', PowerTransformer(), ['balance', 'previous', 'campaign', 'pdays']),
        ('ordinal_encoding', OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']]), ['education'])
    ],
    remainder='passthrough'
)

hgb_pipeline = Pipeline(steps=[
    ('engineer', BankFeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingClassifier(
        random_state=42, 
        categorical_features='from_dtype'
    ))
])

hgb_pipeline.set_output(transform="pandas")
X = train_df.drop('y', axis=1)
y = train_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

hgb_pipeline.fit(X_train, y_train)

joblib.dump(hgb_pipeline, 'model.joblib')
