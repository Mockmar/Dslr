import pandas as pd
import sys

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
        description[column] = {
            'count': df[column].count(),
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            '25%': df[column].quantile(0.25),
            '50%': df[column].median(),
            '75%': df[column].quantile(0.75),
            'max': df[column].max()
        }
    return pd.DataFrame(description)

def describe(df):
    print('General Information')
    print(custom_describe(df.groupby('Hogwarts House')).loc[:, (slice(None), 'std')])
    print('\n')
    house_df = repartion_non_numerical_features(df, 'Hogwarts House')
    print('House Repartition')
    print(house_df.to_string(index=False))
    print('\n')
    print('Best Hand Repartition')
    hand_df = repartion_non_numerical_features(df, 'Best Hand')
    print(hand_df.to_string(index=False))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    path = sys.argv[1]
    df = import_csv('datasets/dataset_train.csv')
    df = preprocesing(df)
    # df = extract_numerical_features(df)
    describe(df)



