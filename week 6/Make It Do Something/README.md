---
title: Uber Fare Predictor
emoji: 🚕
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Uber Fare Predictor — Live Demo

A live demo of the fare-prediction pipeline built during my ML internship at Cellula Technologies:
EDA → preprocessing (ColumnTransformer: StandardScaler + OrdinalEncoder + OneHotEncoder) →
tuned HistGradientBoostingRegressor, saved as a single sklearn pipeline and served here through Gradio.

Enter trip details on the left, get a fare estimate on the right. Runs entirely on Hugging Face's free CPU tier.
