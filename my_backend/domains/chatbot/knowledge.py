"""Curated product knowledge injected (cached) into the chatbot system prompt.

Single source, written in English with key German UI terms in parentheses so the
assistant uses the labels the user sees. Maintained by hand — keep it in sync with
the in-app tooltips (Frontend/src/locales/*/training.json) when parameters change.
Do not invent numeric defaults; cite the in-app tooltip for exact values.
"""

PRODUCT_KNOWLEDGE = """\
# Forecast Engine — product knowledge for the in-app assistant

Forecast Engine is a self-service AI forecasting platform: users upload historical
time-series data, clean and configure it, train a model, and get forecasts (download
the model or call its REST API).

## Workflow (the steps a user moves through)
1. Upload — upload a CSV/TXT time series.
2. Standardisation (Standardisierung) — resample the data onto a uniform time grid.
3. Time-grid / resolution (Zeitrastereinstellung) — choose the time step (Zeitschrittweite)
   and offset of that grid.
4. Anomaly detection (Anomalieerkennung) — remove sensor faults, outliers and gaps.
5. Data adjustments (Datenanpassung) — manual corrections / trimming irrelevant periods.
6. Data cloud (Datenwolke) — fit a regression cloud with confidence bounds.
7. Training — pick a model architecture, set hyperparameters, train, read validation metrics.
8. Forecast / API — download the trained model, or POST new data to its REST endpoint.

## Data preparation (how to get accurate forecasts)
- Use a clean, continuous series with consistent timestamps and a single, regular
  resolution; remove sensor errors and obvious outliers first (anomaly detection step).
- Provide enough history to cover the patterns you want to predict — several full cycles
  of the longest seasonality that matters (e.g. multiple years if there is a yearly cycle).
- Include the driver variables that physically influence the target (e.g. weather for
  energy or heat demand). Forecast Engine can integrate weather-forecast data.
- Do not include information that would not be available at prediction time (no leakage).
  More input variables is not automatically better — prefer informative drivers.

## Choosing input variables (which data to feed the model)
Include the target's real drivers and relevant exogenous signals; exclude redundant or
leaking columns. If unsure, start with the obvious physical drivers and compare models
with and without a candidate variable using the validation metrics.

## Time information (which time features are relevant)
Calendar/time features help the model capture cyclic and seasonal structure. The available
features (configured on the Training step under "Time Information") are:
- Jahr (Year) — position within the calendar year / season.
- Monat (Month) — position within the month.
- Woche (Week) — position within the week, Monday to Sunday (weekday effect).
- Tag (Day) — position within a day, 00:00 to 24:00 (hour-of-day effect).
- Feiertag (Holiday) — public holiday indicator (Germany / Austria / Switzerland;
  note: holidays before 2020 are not supported).
- Zeitzone (Timezone) — required when the "Consideration of daylight saving time"
  (detaillierte Berechnung) option is active; supported zones include UTC, Vienna, and
  Los Angeles. By default all features refer to UTC without DST adjustment.
Include the features whose cycle matches your target (e.g. Tag and Woche for an
electricity-load forecast; Jahr for a seasonal heating forecast). Time features are encoded
as sine/cosine pairs so the model sees a continuous, jump-free representation of periodicity.

## Model selection (which architecture for which problem)
Forecast Engine offers several architectures; train more than one and compare metrics:
- Linear regression (LIN) — fast, interpretable baseline; no extra parameters required.
- Dense (feed-forward neural net) — a strong simple baseline when relationships are mostly
  static and driver variables carry most of the signal.
  Default hyperparameters (from in-app tooltips): layers 3, neurons per layer 512,
  max epochs 20, activation ReLU. Valid ranges: layers 1–10, neurons 1–2048, epochs 1–1000.
  Reference values from tooltips: layers 2–3 (simple) / 3–4 (complex) / >4 rarely needed;
  neurons 16–32 (small) / 32–128 (standard) / 128–512 (complex data).
- CNN — captures local temporal patterns / short-range shape in the series.
  Default hyperparameters: layers 3, filters per layer 512, kernel size 3, max epochs 20,
  activation ReLU. Valid ranges: layers 1–10, kernel size odd 1–11, epochs 1–1000.
  Reference values: layers 2–4 (simple) / 3–6 (complex); filters 16–32 / 32–128 / 128–512;
  kernel size 2–3 (fine, standard) / 3–5 (coarser patterns) / >5 rarely necessary.
- LSTM and AR-LSTM (autoregressive) — sequential models for longer temporal dependencies and
  seasonality; AR-LSTM feeds its own previous predictions forward for multi-step horizons.
  Default hyperparameters: layers 3, neurons per layer 512, max epochs 20, activation ReLU.
  Reference values for layers/neurons: same as Dense above.
- SVR (support vector regression; direct "dir" and MIMO variants) — robust on smaller
  datasets and nonlinear relationships; "dir" predicts each horizon step independently,
  "MIMO" predicts all horizon steps jointly.
  Default hyperparameters: kernel poly, C (regularization) 1, epsilon (tolerance) 0.1.
  Valid ranges: C positive number; epsilon 0–1.
  Reference values: C typically 0.1–10 (default 1; reduce for overfitting, increase for
  coarse predictions); epsilon 0.01–0.1 (higher for noisy data, lower for accurate data).
  Kernel choices: rbf (standard, most applications), linear (simple / quick start),
  poly (known curved relationships), sigmoid (rare).
- LGBMR (LightGBM gradient boosting) — strong general-purpose tabular model; handles many
  input variables well and is a good default when you have rich feature/driver data.
  Default hyperparameters: n_estimators 100, learning_rate 0.1, max_depth -1 (unlimited).
  Reference values from tooltips: n_estimators 50–200 (simple) / 100–500 (typical range);
  learning_rate 0.01–0.1 (good starting value 0.05; smaller values need more trees);
  max_depth 3–6 (simple to medium) / 6–10 (complex) / deeper rarely useful.
All neural network models (Dense, CNN, LSTM, AR-LSTM) are compiled with the Adam optimizer
(learning rate 0.001), MSE loss, and early stopping (patience 2, restore best weights), so
training stops automatically when validation loss stops improving.
Guidance: start with a quick baseline (Linear or LGBMR), then test LSTM/CNN if the series
has strong temporal/seasonal structure.

## Data split (train / validate / test)
The dataset is split into training, validation and test subsets. A split of 70% / 20% / 10%
has proven useful in practice (and is the platform default). The training set optimizes model
parameters; validation monitors generalization during training; the test set gives a final
unbiased performance estimate on previously unseen data. If data shuffle is disabled, the
oldest data is used for training and the most recent for testing.

## Forecast horizon (how far ahead)
The horizon is how far into the future you need predictions, driven by your use case (e.g. a
day-ahead schedule needs 24 hours; intraday control needs a few hours). Longer horizons are
inherently harder and less accurate — only forecast as far ahead as your decision actually
requires.

## Temporal resolution (how fine a time step)
The resolution is the spacing of the forecast points (Zeitschrittweite). Match it to your
decision needs and to your data's native resolution: you cannot reliably forecast at a finer
resolution than your input data supports, and finer resolution costs more data and compute.
Common choices follow the cadence of the process (e.g. 15-minute or hourly for energy).
Non-integer values must be entered as decimals. The offset (Offset) shifts the reference
time grid relative to the full hour (in minutes).

## Scaling and using a downloaded model
Inputs and outputs are scaled before training using per-feature MinMaxScalers, and the fitted
scalers are saved together with the model (downloadable as `.save` files alongside the
`.keras`/`.pkl` model file). To use a downloaded model in your own environment:
1. Load the saved model and the saved scaler(s).
2. Build inputs with the SAME columns, order, and resolution as in training.
3. Apply the saved INPUT scaler to your inputs (transform — do not re-fit).
4. Run inference to get the (scaled) prediction.
5. Apply the saved OUTPUT scaler's inverse transform to convert the prediction back to real units.
Skipping the scaling, re-fitting the scaler, or changing the column order will produce wrong
results. The platform's "Environment Info" endpoint reports the exact Python/TensorFlow/Keras
versions used during training so you can align your local environment before loading a model.

## Validation metrics
Each trained model reports error metrics on a held-out validation split:
- MAE (Mean Absolute Error) — average absolute deviation; same unit as the target; treats all
  errors equally; underweights large rare errors.
- MAPE (Mean Absolute Percentage Error) — percentage average deviation; avoid when true values
  are near zero.
- MSE (Mean Squared Error) — penalizes large errors more; forms the basis of RMSE.
- RMSE (Root Mean Squared Error) — same unit as target; compare with MAE to detect large
  sporadic errors (big RMSE–MAE gap = inconsistent error magnitude).
- NRMSE (Normalized RMSE) — RMSE divided by mean or range; useful for comparing across
  datasets with different scales.
- WAPE (Weighted Average Percentage Error) — weighted to prevent small values distorting the
  average; recommended when data contains very small values.
- sMAPE (symmetric MAPE) — treats over- and under-predictions equally; more robust than MAPE
  when values are small.
- MASE (Mean Absolute Scaled Error) — unit-free; scaled by a naive forecast; comparable
  across different datasets.
Lower error and higher R² are better; compare candidate models on the same validation split
rather than on training data. The metrics panel also supports a delta (aggregation period in
minutes) to compare accuracy at different temporal resolutions.
"""
