# import libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from flask import render_template

# Flask App Initialization
app = Flask(__name__)


# Load datasets
consumer_data = pd.read_csv("Consumer_Dataset.csv", encoding = "latin1")
donation_data = pd.read_csv("Donation_Dataset.csv", encoding = "latin1")

# Preprocess data
donation_data["Day"] = pd.to_datetime(donation_data["Day"])
consumer_data["Visit Date"] = pd.to_datetime(consumer_data["Visit Date"])
monthly_donations = donation_data.groupby(donation_data["Day"].dt.to_period("M"))["Quantity"].sum()
monthly_consumption = consumer_data.groupby(consumer_data["Visit Date"].dt.to_period("M"))["Quantity Taken"].sum()


# Routes for different tabs
# Home
@app.route("/")
def home():
    return render_template("index.html")


# Data Visualization with Descriptions
@app.route("/visualization")
def visualization():
    try:
        # Generate separate subplots for donations and consumption
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Donations
        ax1.plot(monthly_donations.index.to_timestamp(), monthly_donations, label="Monthly Donations", color="blue")
        ax1.set_title("Monthly Donations")
        ax1.set_ylabel("Total Quantity")
        ax1.legend()

        # Consumption
        ax2.plot(monthly_consumption.index.to_timestamp(), monthly_consumption, label="Monthly Consumption", color="orange", linestyle="--")
        ax2.set_title("Monthly Consumption")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Quantity")
        ax2.legend()

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)  # Close the figure to prevent overlap
        buffer.seek(0)
        plot_url_subplots = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()


        # Generate EDA plots and descriptions
        eda_plots = []
        eda_descriptions = []

        # 1. Histogram for donation quantity distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(donation_data["Quantity"], bins=30, color="skyblue", edgecolor="black")
        ax.set_title("Distribution of Donation Quantities")
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        eda_plots.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        eda_descriptions.append("This histogram shows how donation quantities are distributed. It helps identify the most common donation sizes and detect any anomalies or outliers.")
        buffer.close()

        # 2. Box plot for quantities taken in consumer data
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(consumer_data["Quantity Taken"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax.set_title("Box Plot of Quantities Taken by Consumers")
        ax.set_xlabel("Quantity Taken")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        eda_plots.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        eda_descriptions.append("The box plot illustrates the distribution of quantities taken by consumers, highlighting the median, quartiles, and potential outliers.")
        buffer.close()

        # 3. Bar plot for top 5 most popular items in donation data
        top_donated_items = donation_data["Item"].value_counts().head()
        fig, ax = plt.subplots(figsize=(10, 5))
        top_donated_items.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
        ax.set_title("Top Donated Items")
        ax.set_xlabel("Item")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        eda_plots.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        eda_descriptions.append("This bar chart displays the top 5 items donated, showing which items are most frequently contributed to the pantry.")
        buffer.close()

        # 4. Bar plot for top 5 most popular items in consumer data
        top_requested_items = consumer_data["Item Taken"].value_counts().head()
        fig, ax = plt.subplots(figsize=(10, 5))
        top_requested_items.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
        ax.set_title("Top Requested Items")
        ax.set_xlabel("Item")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        eda_plots.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        eda_descriptions.append("This bar chart shows the top 5 items requested by consumers, indicating the most needed items.")
        buffer.close()
        
        # EDA plot titles
        eda_titles = [
        "Distribution of Donation Quantities",
        "Box Plot of Quantities Taken by Consumers",
        "Top Donated Items",
        "Top Requested Items"
    ]
        # Render the visualization template with the plots, descriptions and titles
        return render_template(
        "visualization.html",
        plot_url_subplots=plot_url_subplots,
        eda_plots=eda_plots,
        eda_descriptions=eda_descriptions,
        eda_titles=eda_titles
    )

    except Exception as e:
        return f"Error in Visualization: {e}"




# Forecasting
@app.route("/forecasting", methods=["GET", "POST"])
def forecasting():
    try:
        # Initialize variables for n_lags and plot URLs
        n_lags = 12
        donation_plot_url = None
        consumer_plot_url = None

        if request.method == "POST":
            # Get n_lags value from the form
            n_lags = int(request.form.get("n_lags", 12))

            # Function to create lagged features
            def create_lagged_features(data, n_lags):
                X, y = [], []
                for i in range(n_lags, len(data)):
                    X.append(data[i - n_lags:i])  # Use previous `n_lags` months as features
                    y.append(data[i])           # Current data point is the target
                return np.array(X), np.array(y)

            ### Donation Forecast ###
            donation_data = monthly_donations.values
            X_donation, y_donation = create_lagged_features(donation_data, n_lags)
            X_train, X_test, y_train, y_test = train_test_split(X_donation, y_donation, test_size=0.2, random_state=42)
            donation_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            donation_model.fit(X_train, y_train)

            # Forecast for the next 24 months
            donation_predictions = []
            current_input = donation_data[-n_lags:]
            for _ in range(24):
                pred = donation_model.predict(current_input.reshape(1, -1))[0]
                donation_predictions.append(pred)
                current_input = np.append(current_input[1:], pred)

            donation_forecast_index = pd.date_range(start=monthly_donations.index[-1].to_timestamp(), periods=24, freq="M")

            # Plot donation forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(monthly_donations.index.to_timestamp(), monthly_donations, label="Historical Donations")
            ax.plot(donation_forecast_index, donation_predictions, label="Forecasted Donations", linestyle="--")
            ax.set_title("Donation Forecast")
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Quantity")
            ax.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            donation_plot_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
            buffer.close()

            ### Consumer Forecast ###
            consumer_data = monthly_consumption.values
            X_consumer, y_consumer = create_lagged_features(consumer_data, n_lags)
            X_train, X_test, y_train, y_test = train_test_split(X_consumer, y_consumer, test_size=0.2, random_state=42)
            consumer_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            consumer_model.fit(X_train, y_train)

            # Forecast for the next 24 months
            consumer_predictions = []
            current_input = consumer_data[-n_lags:]
            for _ in range(24):
                pred = consumer_model.predict(current_input.reshape(1, -1))[0]
                consumer_predictions.append(pred)
                current_input = np.append(current_input[1:], pred)

            consumer_forecast_index = pd.date_range(start=monthly_consumption.index[-1].to_timestamp(), periods=24, freq="M")

            # Plot consumer forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(monthly_consumption.index.to_timestamp(), monthly_consumption, label="Historical Consumption")
            ax.plot(consumer_forecast_index, consumer_predictions, label="Forecasted Consumption", linestyle="--")
            ax.set_title("Consumer Forecast")
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Quantity")
            ax.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            consumer_plot_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
            buffer.close()

        return render_template(
            "forecasting.html",
            n_lags=n_lags,
            donation_plot_url=donation_plot_url,
            consumer_plot_url=consumer_plot_url
        )

    except Exception as e:
        return f"Error in Forecasting: {e}"


# About
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)