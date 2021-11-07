import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go


DATE_FORMAT = "%Y-%m-%d"
START = "2015-01-01"
TODAY = datetime.date.today().strftime(DATE_FORMAT)


# Main app
st.title("Stock prediction app")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
select_stocks = st.selectbox("Select stock for prediction", stocks)

@st.cache # now don't have to re-download on changing data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data: pd.DataFrame = load_data(select_stocks)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())


def plot_raw_data(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.Date, y=data.Open, name="stock_open"))
    fig.add_trace(go.Scatter(x=data.Date, y=data.Close, name="stock_close"))
    fig.layout.update(title_text="Time Series Data")

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="year start",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    st.plotly_chart(fig)

plot_raw_data(data)


# Forcasting
df_train: pd.DataFrame = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

st.subheader("Forecast data")
num_years = st.slider("Years of prediction", 1, 5)
DURATION = num_years * 365 # FIXME:

m = Prophet()
m.fit(df_train)
future: pd.DataFrame = m.make_future_dataframe(periods=DURATION)
forecast = m.predict(future)

st.write(forecast.tail())

st.write("Forecast Data")
fig_plotly = plot_plotly(m, forecast)
st.plotly_chart(fig_plotly)

st.write("Forecast components")
fig_comp = m.plot_components(forecast)
st.write(fig_comp)


