import sys
import pickle
import pandas as pd

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context


def read_data(taxi_type, year_run, month_run):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year_run:04d}-{month_run:02d}.parquet'
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df['ride_id'] = f'{year_run:04d}/{month_run:02d}_' + df.index.astype('str')
    return df

def load_model(model_file):
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def prep_dict(df):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def save_result(taxi_type, year_run, month_run, df, y_pred):
    output_file = f'./output/{taxi_type}/{year_run:04d}-{month_run:02d}.parquet'
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return output_file

@task
def apply_model(taxi_type, year_run, month_run, model_file):
    logger = get_run_logger()

    logger.info(f'reading the data for {year_run} {month_run}...')
    df = read_data(taxi_type, year_run, month_run)
    dicts = prep_dict(df)
    logger.info(f'loading the model with model file = {model_file}...')
    dv, model = load_model(model_file)
    
    logger.info(f'applying the model...')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f'average of y_pred = {y_pred.mean()}')
    
    logger.info(f'saving the result...')
    output_file = save_result(taxi_type, year_run, month_run, df, y_pred)
    
    return output_file

@flow
def run():
    taxi_type = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3
    model_file = sys.argv[4] # 'model.bin'

    apply_model(taxi_type, year_run = year, month_run = month, model_file = model_file)


if __name__ == '__main__':
    run()