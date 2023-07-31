import matplotlib.pyplot as plt
import pandas as pd
import datetime
from io import BytesIO
import base64
from flask import render_template, request, make_response
import yfinance as yf
from src import app
from src.utilities import MasterProphet
from src.utilities import Dataset

@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response

@app.route("/")
@app.route("/home")
def home():
    """ Renders the home page """
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    ticker = request.form["ticker"]
    master_prophet = MasterProphet(ticker)

    forecast = master_prophet.forecast()

    # Fetch historical data using yfinance
    historical_data = yf.download(ticker, start='2010-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'))

    actual_forecast = round(forecast.yhat[0], 2)
    lower_bound = round(forecast.yhat_lower[0], 2)
    upper_bound = round(forecast.yhat_upper[0], 2)
    bound = round(((upper_bound - actual_forecast) + (actual_forecast - lower_bound) / 2), 2)

    summary = master_prophet.info["summary"]
    country = master_prophet.info["country"]
    sector = master_prophet.info["sector"]
    website = master_prophet.info["website"]
    min_date = master_prophet.info["min_date"]
    max_date = master_prophet.info["max_date"]

    forecast_date = master_prophet.forecast_date.date()

    # Plotting the forecast using fbprophet's plot function
    fig, ax = plt.subplots(figsize=(12, 8))
    master_prophet.model.plot(forecast, ax=ax)
    ax.set_title("Forecast for Stock Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)

    # Add markers for actual and forecasted values
    ax.plot(historical_data.index[-365:], historical_data["Close"].iloc[-365:], label='Actual')
    ax.plot(forecast_date, actual_forecast, 'ro', markersize=8, alpha=0.5, label='Forecast')

    # Add legend and grid lines
    ax.legend()
    ax.grid(True)

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight', dpi=300)
    buffer.seek(0)

    # Encode the plot image to base64
    plot_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    plt.close()

    # Create response with the rendered template
    response = make_response(render_template(
        "output.html",
        ticker=ticker.upper(),
        sector=sector,
        country=country,
        website=website,
        summary=summary,
        min_date=min_date,
        max_date=max_date,
        forecast_date=forecast_date,
        forecast=actual_forecast,
        bound=bound,
        plot_image=plot_image
    ))

    # Add necessary headers for the image
    response.headers['Content-Type'] = 'text/html'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response
