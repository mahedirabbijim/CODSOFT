
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]

def make_text_pipeline(key):
    return Pipeline([
        ('selector', TextSelector(key=key)),
        ('vectorizer', CountVectorizer())
    ])

df = pd.read_csv('IMDb_Movies_India_Cleaned.csv')
df = df.dropna(subset=['Rating'])
text_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
df[text_cols] = df[text_cols].fillna('Unknown')
X = df[text_cols]
y = df['Rating'].astype(float)

preprocessor = FeatureUnion([
    ('genre', make_text_pipeline('Genre')),
    ('director', make_text_pipeline('Director')),
    ('actor1', make_text_pipeline('Actor 1')),
    ('actor2', make_text_pipeline('Actor 2')),
    ('actor3', make_text_pipeline('Actor 3'))
])

model_pipeline = Pipeline([
    ('features', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)
print("Model Test Score:", model_pipeline.score(X_test, y_test))
