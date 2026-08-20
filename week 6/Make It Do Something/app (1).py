"""
Live fare-prediction demo — Week 8 "Make It Do Something"

WHAT TO DO BEFORE DEPLOYING (2 steps, ~5 minutes):

1) Put your real trained pipeline file next to this script, named:
   fare_amount_pipeline.joblib

2) Find out EXACTLY which raw columns your pipeline expects, and in
   what order. Run this in a notebook or terminal:

       import joblib
       pipeline = joblib.load("fare_amount_pipeline.joblib")
       print(list(pipeline.feature_names_in_))

   Then edit the FEATURE_COLUMNS list and the matching Gradio inputs
   below so they match EXACTLY (same names, same order). Everything
   marked "ADJUST ME" is a placeholder guess based on what's in your
   Task 2 notes (Car Condition / Weather Condition / Traffic Condition
   + distance-based features) — swap it for your real columns.
"""

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = "fare_amount_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)

# ADJUST ME — must match pipeline.feature_names_in_ exactly, same order.
FEATURE_COLUMNS = [
    "distance_km",
    "passenger_count",
    "hour",
    "Car Condition",
    "Weather",
    "Traffic Condition",
]


def predict_fare(distance_km, passenger_count, hour, car_condition, weather, traffic):
    """Builds one row matching the pipeline's expected schema and predicts."""
    row = pd.DataFrame(
        [[distance_km, passenger_count, hour, car_condition, weather, traffic]],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction = pipeline.predict(row)[0]
        return f"Estimated fare: ${prediction:.2f}"
    except Exception as e:
        # Fails gracefully instead of crashing the page — required by the brief.
        return f"Couldn't price this trip ({e}). Check that the inputs match the model's expected columns."


with gr.Blocks(title="Uber Fare Predictor — Live Demo") as demo:
    gr.Markdown(
        "## Uber Fare Predictor\n"
        "A live demo of the regression pipeline built during my Cellula Technologies ML internship "
        "(EDA → preprocessing → HistGradientBoostingRegressor, deployed here from a saved sklearn pipeline)."
    )
    with gr.Row():
        with gr.Column():
            distance_km = gr.Number(label="Trip distance (km)", value=5.0)
            passenger_count = gr.Number(label="Passenger count", value=1, precision=0)
            hour = gr.Slider(0, 23, value=12, step=1, label="Hour of day (0–23)")
            # ADJUST ME — swap these choices for the real categories your OrdinalEncoder/OneHotEncoder were fit on.
            car_condition = gr.Dropdown(
                ["Excellent", "Very Good", "Good", "Bad", "Very Bad"],
                value="Good",
                label="Car condition",
            )
            weather = gr.Dropdown(
                ["Sunny", "Cloudy", "Rainy", "Windy", "Stormy"],
                value="Sunny",
                label="Weather",
            )
            traffic = gr.Dropdown(
                ["Flow Traffic", "Congested Traffic", "Dense Traffic"],
                value="Flow Traffic",
                label="Traffic condition",
            )
            submit = gr.Button("Predict fare", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Result", interactive=False)

    submit.click(
        predict_fare,
        inputs=[distance_km, passenger_count, hour, car_condition, weather, traffic],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
