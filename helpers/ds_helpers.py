import pandas as pd
import numpy as np
import sys
import dotenv
import os


def open_and_prepare_df(ds):

    dotenv.load_dotenv()
    main_dir = os.getenv('MAINDIR')

    path = f'{main_dir}datasets/'
    if ds == 'eval':
        df = pd.read_excel(io=f'{path}NLP_PILOT.xlsx')

    elif ds == 'main':
        df = pd.read_excel(io=f'{path}NLP_CLEAN.xlsx')
        df = df[df['time'] > 300]
        df = df[df['label'] != 1]

    elif ds == 'features':
        df = pd.read_excel(io=f'{path}NLP_FEATURES.xlsx')

    for item in [f'nlp_{i}' for i in range(2, 6)]:
        df = df[df[item].apply(lambda x: len(x) > 10)]

    return df

def X_y_split(df, X_string):
    
    X = []
    X_raw = df[X_string].values
    for item in X_raw:
        X.append(eval(item))

    X = np.asarray(X)
    y = df['label'].values

    return X, y
