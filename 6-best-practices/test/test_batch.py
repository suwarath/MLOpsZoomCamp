import pandas as pd
from datetime import datetime
import batch
import unittest

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

class TestPrepareData(unittest.TestCase):
    def test_prep_data(self):
        data = [
            (None, None, dt(1, 1), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, None, dt(1, 2, 0), dt(1, 2, 59)),
            (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
        ]

        columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
        df = pd.DataFrame(data, columns=columns)
        
        categorical = ['PULocationID', 'DOLocationID']
        
        df_actual = batch.prepare_data(df, categorical)
        
        df_expected = [
            ('-1','-1',9.0),
            ('1','1',8.0)
        ]
        
        columns_test = ['PULocationID', 'DOLocationID','duration']
        df_expected = pd.DataFrame(df_expected, columns=columns_test)
        
        print("Number of rows in the expected dataframe:", df_expected.shape[0])
        
        self.assertTrue(df_actual['PULocationID'].equals(df_expected['PULocationID']))
        self.assertTrue(df_actual['DOLocationID'].equals(df_expected['DOLocationID']))
        self.assertTrue(df_actual['duration'].equals(df_expected['duration']))
        
if __name__ == '__main__':
    unittest.main()
        