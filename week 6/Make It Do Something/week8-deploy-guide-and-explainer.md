# Week 8 — "Make It Do Something": Live Fare-Prediction Demo

Feature chosen: a live, working demo of the Cellula fare-prediction model (per the brief's own ML-track note: *"proof that runs beats proof you describe"*).

I built and smoke-tested the app logic here with a stand-in model matching your Task 2 schema — prediction and graceful-failure paths both work. You need to swap in your real model before this is genuinely "yours," which takes about 5 minutes.

---

## Step 1 — Match the app to your real pipeline (do this first)

In a notebook, next to your real `fare_amount_pipeline.joblib`:

```python
import joblib
pipeline = joblib.load("fare_amount_pipeline.joblib")
print(list(pipeline.feature_names_in_))
```

Open `app.py` and find the two spots marked `# ADJUST ME`:
1. `FEATURE_COLUMNS` — replace with the exact list you just printed, same order.
2. The `gr.Dropdown(...)` choices for Car Condition / Weather / Traffic — replace with the exact category strings your encoders were fit on (check `pipeline.named_steps[...]` or your training notebook if unsure).

If your pipeline uses different or additional raw features (e.g. pickup/dropoff coordinates instead of a precomputed distance), add matching `gr.Number`/`gr.Slider` inputs and extend `predict_fare`'s parameter list — the structure stays the same either way.

## Step 2 — Test it locally

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL it prints, submit a real trip, confirm you get back a sane fare number — not an error. This is your proof it "genuinely works" before it goes live.

## Step 3 — Deploy on Hugging Face Spaces (free tier)

1. Create a free account at huggingface.co
2. Click **New Space** → name it → SDK: **Gradio** → hardware: **CPU basic (free)**
3. Upload four files into the Space: `app.py`, `requirements.txt`, `README.md`, and your real `fare_amount_pipeline.joblib`
4. Wait for the build log to finish (1–3 minutes) — the Space auto-launches `app.py`
5. Open the live Space URL, submit a real test trip, **screenshot the result** — this screenshot is your evidence for the portal

## Step 4 — Link it from your portfolio

Add the Space URL as the primary CTA on your Cellula case-study page (Week 3 content map already has this slot — "Deployment (Flask app + screenshots)" section).

## Step 5 — Submit

**Deliverable links:** your Hugging Face Space URL
**Files:** your screenshot of a real prediction
**Notes:** paste the explainer below (in your own words if you rephrase it — the brief specifically checks it's *your* understanding)

---

## Plain-words explainer (draft — reword to sound like you)

**What a backend is:** the part of an app that does or remembers something a plain web page can't. My page can't run a machine-learning model by itself — a browser only knows how to display text and take clicks. So there has to be a computer somewhere else, always on, that actually holds the trained model and does the math. That's the backend.

**What my feature does:** it lets a visitor type in trip details — distance, passenger count, time of day, car condition, weather, traffic — and get back a real fare estimate from the actual model I built and tuned during my Cellula internship. It's not a mockup number; it's the same `HistGradientBoostingRegressor` pipeline from Task 2, just running live instead of sitting in a notebook.

**How the data flows:**
1. The visitor fills in the form in their browser (this is Gradio's frontend — just HTML/JS Gradio generates for me).
2. When they click "Predict fare," the browser sends those values to a Python function running on Hugging Face's server — this is the backend, and it's the same server the whole time the Space is live, not something spun up per request.
3. That server already has my saved pipeline loaded in memory (`fare_amount_pipeline.joblib`) — it loads once when the Space starts, not on every click.
4. The function builds one row of data in the exact shape the pipeline was trained on, and calls `pipeline.predict()`.
5. The number that comes back is sent back over the same connection and displayed in the visitor's browser.

No database, no email, nothing gets stored — it's a pure request-in, prediction-out backend. That's the whole loop.
