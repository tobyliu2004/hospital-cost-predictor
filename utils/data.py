import pandas as pd
import numpy as np

#converts a list of columns in df to datetime format
#you have to pass a list of the names of columns you want to convert as a parameter
#if you have a column that is not supposed to be date time it will cause problems
def convert_dates(df, date_cols):
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

#converts length of stay column into a numeric column
#for values like N/A, unknown, they become NaN
def clean_length_of_stay(df):
    df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')
    return df

#creates a log transformed version of total charges called log total charge 
def log_transform_target(df, target_col='Total Charges'):
    df['Log Total Charges'] = df[target_col].apply(lambda x: np.log1p(x))
    return df