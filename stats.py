import pandas as pd
import numpy as np

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