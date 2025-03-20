import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class Preprocess:
    def __init__(self, df=None, normalizer=None, encoders=None):
        self.encoders = encoders or {}
        self.initial_df = df
        self.normalizer = normalizer or StandardScaler()

    def preprocesing(self, df):
        df.dropna(axis=0, inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['Index'], inplace=True)
        return df

    def calculate_age(self, df, today_year):
        df['Birthday'] = pd.to_datetime(df['Birthday'])
        df['Age'] = df['Birthday'].dt.year
        df['Age'] = df['Age'].apply(lambda x: today_year - x)
        return df
    
    def encode_column(self, df, column):
        self.encoders = {column: LabelEncoder()}
        df[column] = self.encoders[column].fit_transform(df[column])
        return df
    
    def encode_column_predict(self, df, column):
        if column not in self.encoders:
            raise ValueError(f'Encoder for column {column} not found')
        df[column] = self.encoders[column].transform(df[column])
        return df
    
    def decode_column(self, df, column):
        df[column] = self.encoders[column].inverse_transform(df[column])
        return df
    
    def normalize(self, df):
        return self.normalizer.fit_transform(df)
    
    def inverse_normalize(self, df):
        return self.normalizer.inverse_transform(df)
    
    def fit(self, df):
        self.initial_df = df
        self.preprocesing(df)
        self.calculate_age(df, 2020)
        self.encode_column(df, 'Best Hand')
        target = df['Hogwarts House']
        features = df.select_dtypes(include=['float64', 'int64'])
        features = self.normalize(features)
        return features, target
    
    def fit_predict(self, df):
        df.drop(columns=['Hogwarts House'], inplace=True)
        self.preprocesing(df)
        self.calculate_age(df, 2020)
        self.encode_column_predict(df, 'Best Hand')
        features = df.select_dtypes(include=['float64', 'int64'])
        features = self.normalizer.transform(features)
        return features
    
    def export_encoders(self, path, column):
        joblib.dump(self.encoders[column], path)

    def export_normalizer(self, path):
        joblib.dump(self.normalizer, path)

