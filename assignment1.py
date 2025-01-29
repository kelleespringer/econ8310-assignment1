import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Step 1: Load the training and test data from the provided URLs
train_url = 'https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv'
test_url = 'https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv'

train_data = pd.read_csv(train_url, parse_dates=['Timestamp'], index_col='Timestamp')
test_data = pd.read_csv(test_url, parse_dates=['Timestamp'], index_col='Timestamp')

# Step 2: Focus on the 'number of trips' column from the training dataset
train_data = train_data['trips']

# Step 3: Define and fit the Exponential Smoothing model
model = ExponentialSmoothing(train_data, 
                             trend='add',  # You can change to 'mul' for multiplicative trend if needed
                             seasonal='add',  # Change to 'mul' if the data is multiplicative
                             seasonal_periods=24*7  # Weekly seasonality, assuming data has a weekly pattern
                            )

# Fit the model
modelFit = model.fit()

# Step 4: Forecast for the next 744 hours (January data from the test set)
forecast_period = 744
pred = modelFit.forecast(forecast_period)

# Step 5: Prepare data for plotting with Plotly
forecast_dates = pd.date_range(train_data.index[-1], periods=forecast_period+1, freq='H')[1:]
forecast_df = pd.DataFrame({'Timestamp': forecast_dates, 'forecasted_trips': pred})

# Combine training data and forecasted data for visualization
train_df = train_data.reset_index()
train_df.columns = ['Timestamp', 'number_of_trips']
forecast_df.columns = ['Timestamp', 'forecasted_trips']

# Concatenate the training and forecasted data
full_df = pd.concat([train_df, forecast_df])

# Step 6: Plotting the data using Plotly Express
fig = px.line(full_df, x='Timestamp', y=['number_of_trips', 'forecasted_trips'], 
              labels={'Timestamp': 'Date', 'value': 'Number of Trips'},
              title='Exponential Smoothing Forecast vs Actual Data')

fig.show()

# Optional: Display first few rows of forecasted data
print(forecast_df.head())
