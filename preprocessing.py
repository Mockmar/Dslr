import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from stats import custom_mean, custom_std

class CustomLabelEncoder:
    def __init__(self):
        self.mapping = {}
        self.inverse_mapping = {}
    
    def fit(self, data):
        unique_values = np.unique(data)
        for i, value in enumerate(unique_values):
            self.mapping[value] = i
            self.inverse_mapping[i] = value
    
    def transform(self, data):
        return np.array([self.mapping[value] for value in data])
    
    def inverse_transform(self, data):
        return np.array([self.inverse_mapping[value] for value in data])
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def save(self, filename):
        df = pd.DataFrame(self.mapping.items(), columns=['value', 'label'])
        df.to_csv(filename, index=False)

    def load(self, filename):
        df = pd.read_csv(filename)
        self.mapping = dict(zip(df['value'], df['label']))
        self.inverse_mapping = dict(zip(df['label'], df['value']))

class CustomStandardScaler:
    def __init__(self):
        self.data_dict = {'column': [], 'mean': [], 'std': []}

    def fit(self, df, except_columns=[]):
        """Calcule la moyenne et l'écart-type pour chaque colonne de X"""
        numerial_columns = df.select_dtypes(include=['Int64', 'float64']).columns
        numerial_columns = [col for col in numerial_columns if col not in except_columns]
        for col in numerial_columns:
            self.data_dict['column'].append(col)
            self.data_dict['mean'].append(custom_mean(df[col]))
            self.data_dict['std'].append(custom_std(df[col]))
        return self
    
    def transform(self, df):
        """Applique la transformation standardisée"""
        for i in range(len(self.data_dict['column'])):
            col = self.data_dict['column'][i]
            mean = self.data_dict['mean'][i]
            std = self.data_dict['std'][i]
            df.loc[:,col] = (df[col] - mean) / std
        return df
    
    def fit_transform(self, df, except_columns=[]):
        """Combine fit() et transform()"""
        return self.fit(df, except_columns).transform(df)
    
    def save(self, filename):
        """Sauvegarde les paramètres dans un fichier texte"""
        df = pd.DataFrame(self.data_dict)
        df.to_csv(filename, index=False)
        
    def load(self, filename):
        """Charge les paramètres depuis un fichier texte"""
        df = pd.read_csv(filename)
        self.data_dict = df.to_dict(orient='list')

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
    