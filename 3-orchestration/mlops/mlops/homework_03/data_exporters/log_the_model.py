import os
from typing import Dict, Optional, Tuple, Union

import mlflow
import pandas as pd


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def log_model(models):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-yellow-taxi-experiment")
    mlflow.sklearn.log_model(models[0], artifact_path="models")
    mlflow.sklearn.log_model(models[1], artifact_path="models")