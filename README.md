# UK Fuel & EV-Aware Route Planner

A Python/Flask prototype for planning UK journeys with petrol/diesel refuelling or EV charging considerations, map-based route simulation and optional machine-learning route scoring.

## Features
- Route planning using Google Maps APIs
- Mapbox-based 3D route visualisation and vehicle HUD
- Fuel and EV range estimation and suggested stops
- Optional ML training and route-quality scoring
- Experimental and earlier implementations retained in `versions/`

## Setup

Requires Python 3.10+ (recommended).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set your own valid keys for `GOOGLE_MAPS_API_KEY` and `MAPBOX_TOKEN`. Enable the necessary Google Maps APIs and configure billing/restrictions as required. Never commit `.env`.

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser. This is a local development setup, not a production deployment.

## Machine learning

The app includes ML routes/endpoints and `train_ml.py` for offline model training. Training needs usable labelled observations in `ml_training_data.jsonl` (which is not included). A trained `ml_route_model.pkl` is also not included. Some scoring functions fall back to baseline/neutral results when no model is available.

## Project structure

```text
app.py                   Main Flask application (selected from uploaded app version)
train_ml.py              Offline model training utility
requirements.txt         Python dependencies
.env.example             API key placeholders
.gitignore               Ignores secrets, local outputs and model files
examples/                Standalone map prototype and UI reference
versions/                Alternate development versions (not required to run app.py)
```

## Security and limitations

- The uploaded source included credentials in a separate env file and a browser-side Mapbox token. These credentials have been omitted/replaced in this repository. **Rotate the Google Maps API key and restrict both providers' tokens** before publishing, especially if they were previously shared.
- Restrict the Google key to the required APIs and an appropriate application or server identity. Mapbox public browser tokens are intended to be client-visible, but should have URL/domain restrictions where appropriate.
- Flask debug is disabled by default in the packaged main app. Do not expose development endpoints publicly without reviewing authentication and rate limits.
- Existing source and alternatives are provided as a portfolio/prototype; functionality requires provider API access and has not been end-to-end verified against live APIs.

## License

No license has been selected. All rights reserved unless a license is explicitly added.
