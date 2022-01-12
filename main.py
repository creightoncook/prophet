# Forecasting in Python
# Step 1: Load raw data
# Step 2: Clean data
# Step 3: Forecast
# Step 4: Report data
import pandas as pd
from prophet import Prophet
import random 

def random_generator(min, max, count):
    rand = []
    for c in range(count):
        rand.append(random.randint(min,max))
    return rand

def load_data():
    '''Generate mock data'''
    return pd.DataFrame({
        "case_count": [
            "10",
            "100",
            "200",
            "300",
            "400",
            "1000"
        ],
        "date": [
            "2021-01-02",
            "2021-01-03",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
        ],
        "tags": [
            "dfsdf1",
            "3sdfsdf5",
            "2234",
            "346sdfdsf5346",
            "234234",
            "34sdfsdf5345"
        ]
    })

def clean_data(df: pd.DataFrame): 
    new_df = df.rename(columns={"case_count": "y","date": "ds"})
    return new_df[["ds","y"]]

def forecast(df: pd.DataFrame):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=5)
    return m.predict(future)

def main():
    df = load_data()
    df = clean_data(df)
    results = forecast(df)
    print(results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__=="__main__":
    '''This code block executes whenever the script is executed from the command line.'''
    main()

