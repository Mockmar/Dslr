import pandas as pd

def import_csv(path):
    return pd.read_csv(path)

def preprocesing(df):
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['Index'], inplace=True)
    return df

def extract_numerical_features(df):
    return df.select_dtypes(include=['float64', 'int64'])

def calculate_age(df, today_year):
    df['Birthday'] = pd.to_datetime(df['Birthday'])
    df['Age'] = df['Birthday'].dt.year
    return df['Age'].apply(lambda x: today_year - x)

def 



