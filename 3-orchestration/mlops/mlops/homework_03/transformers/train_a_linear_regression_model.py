from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train_lr(df: pd.DataFrame) -> Tuple[DictVectorizer, LinearRegression]:
    # Vectorize features
    trn_dict = df[['PULocationID','DOLocationID']].astype({'PULocationID':str,'DOLocationID':str }).to_dict(orient = 'records')
    dv = DictVectorizer(sparse = True)
    dv.fit(trn_dict)
    X_trn = dv.transform(trn_dict)
    Y_trn = df['duration'].values
    # Train Linear Regression Model
    reg = LinearRegression()
    reg.fit(X_trn, Y_trn)
    print(reg.intercept_)

    return dv, reg