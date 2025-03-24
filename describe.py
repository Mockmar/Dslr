import pandas as pd
import numpy as np
from preprocessing import Preprocess
from stats import custom_count, custom_mean, custom_std, custom_min, custom_max, custom_percentile
import sys

def import_csv(path):
    return pd.read_csv(path)

def mean_numerical_feature_per_house(df):
    numerical_columns = df.select_dtypes(include=['float64','int64']).columns.tolist()
    numerical_columns.append('Hogwarts House')
    tmp = df[numerical_columns]
    df_grpby = tmp.groupby('Hogwarts House').mean()
    return df_grpby

def repartion_non_numerical_features(df, column):
    tmp_df = df[column].value_counts()
    tmp_df['total'] = tmp_df.sum()
    count = tmp_df.values.astype(int)
    house = tmp_df.index.tolist()
    percent = (count/count[-1] * 100).round(2)

    data = {
        'Statistic': ['Count', 'Percentage'],
    }

    for i in range(len(house)):
        data[house[i]] = [count[i], percent[i]]

    df_repartition = pd.DataFrame(data)
    return df_repartition

def custom_describe(df):
    description = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        series = df[column]
        description[column] = {
            'count': custom_count(series),
            'mean': custom_mean(series),
            'std': custom_std(series),
            'min': custom_min(series),
            '25%': custom_percentile(series, 0.25),
            '50%': custom_percentile(series, 0.50),
            '75%': custom_percentile(series, 0.75),
            'max': custom_max(series)
        }
    return pd.DataFrame(description)

def describe(df):
    print('---General Information---\n')
    description = custom_describe(df)
    print(description.to_string(index=True))
    print('\n')
    print('---House Repartition---\n')
    house_df = repartion_non_numerical_features(df, 'Hogwarts House')
    print(house_df.to_string(index=False))
    print('\n')
    print('---Mean score per house---\n')
    means = mean_numerical_feature_per_house(df)
    print(means.to_string(index=True))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    preprocess = Preprocess()
    path = sys.argv[1]
    df = import_csv(path)
    df = preprocess.preprocesing(df)
    df = preprocess.calculate_age(df, 2020)
    describe(df)



