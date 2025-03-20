import pandas as pd
import numpy as np
from preprocessing import Preprocess
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

def custom_count(series):
    count = 0
    for x in series:
        if not pd.isnull(x):
            count += 1
    return count

def custom_mean(series):
    count = custom_count(series)
    total = 0
    for x in series:
        if not pd.isnull(x):
            total += x
    return total / count if count != 0 else np.nan

def custom_std(series):
    mean = custom_mean(series)
    count = custom_count(series)
    variance = 0
    for x in series:
        if not pd.isnull(x):
            variance += (x - mean) ** 2
    variance /= (count - 1) if count > 1 else np.nan
    return np.sqrt(variance)

def custom_min(series):
    min_val = None
    for x in series:
        if not pd.isnull(x):
            if min_val is None or x < min_val:
                min_val = x
    return min_val

def custom_max(series):
    max_val = None
    for x in series:
        if not pd.isnull(x):
            if max_val is None or x > max_val:
                max_val = x
    return max_val

def custom_percentile(series, percentile):
    sorted_series = sorted(x for x in series if not pd.isnull(x))
    k = (len(sorted_series) - 1) * percentile
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return sorted_series[int(k)]
    d0 = sorted_series[int(f)] * (c - k)
    d1 = sorted_series[int(c)] * (k - f)
    return d0 + d1

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
    print(description.to_string(index=False))
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



