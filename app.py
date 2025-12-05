import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Environmental ML Dashboard", layout="wide")

st.title("🌍 Environmental Machine Learning Dashboard")
st.write("Predict **Air Quality (AQI Forecast)** and **Water Potability** using trained ML models.")

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2 = st.tabs(["🌫️ Air Quality Forecast", "💧 Water Potability Prediction"])

# ============================================================
# --------------- TAB 1: AIR QUALITY FORECAST ----------------
# ============================================================
with tab1:
    st.header("🌫️ Air Quality Forecasting Dashboard")

    @st.cache_resource
    def load_air_models():
        model_paths = {
            "GradientBoosting": "GradientBoosting.pkl",
            "DecisionTree": "DecisionTree.pkl",
            "RandomForest": "RandomForest.pkl",
            "KNN": "KNN.pkl"
        }

        models = {}
        for name, path in model_paths.items():
            try:
                models[name] = joblib.load(path)
            except:
                st.warning(f"⚠️ Missing model file: {path}")

        scaler = joblib.load("scaler.joblib")
        return models, scaler

    air_models, air_scaler = load_air_models()

    # Load dataset
    df = pd.read_csv("air_quality.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
    df = df.sort_values("date")
    df["aqi_clean"] = df["aqi"]

    # Feature Engineering
    def prepare_air_features(data):
        for lag in [1, 3, 6, 12, 24, 48, 72]:
            data[f"lag_{lag}"] = data["aqi_clean"].shift(lag)

        data["roll_3"] = data["aqi_clean"].rolling(3).mean()
        data["roll_12"] = data["aqi_clean"].rolling(12).mean()
        data["roll_24"] = data["aqi_clean"].rolling(24).mean()
        data["roll_72"] = data["aqi_clean"].rolling(72).mean()

        data["hour"] = data["date"].dt.hour
        data["day"] = data["date"].dt.day
        data["month"] = data["date"].dt.month
        data["weekday"] = data["date"].dt.weekday

        return data.dropna()

    df = prepare_air_features(df)

    FEATURES = [
        "lag_1","lag_3","lag_6","lag_12","lag_24","lag_48","lag_72",
        "roll_3","roll_12","roll_24","roll_72",
        "hour","day","month","weekday"
    ]

    # Forecast logic
    def forecast_future(model, hours):
        future = df.copy()

        for _ in range(hours):
            X = future.iloc[-1][FEATURES].values.reshape(1, -1)
            pred = model.predict(X)[0]

            new_row = {
                "date": future.iloc[-1]["date"] + pd.Timedelta(hours=1),
                "aqi_clean": pred
            }

            for lag in [1, 3, 6, 12, 24, 48, 72]:
                new_row[f"lag_{lag}"] = future["aqi_clean"].iloc[-lag]

            for w in [3, 12, 24, 72]:
                new_row[f"roll_{w}"] = future["aqi_clean"].iloc[-w:].mean()

            new_row["hour"] = new_row["date"].hour
            new_row["day"] = new_row["date"].day
            new_row["month"] = new_row["date"].month
            new_row["weekday"] = new_row["date"].weekday()

            future = pd.concat([future, pd.DataFrame([new_row])], ignore_index=True)

        return future.tail(hours)[["date", "aqi_clean"]]

    st.sidebar.header("⏳ Air Forecast Settings")
    forecast_choice = st.sidebar.selectbox(
        "Select Prediction Period",
        ["Next 7 Days", "Next 7 Weeks", "Next 30 Days"]
    )

    if forecast_choice == "Next 7 Days":
        horizon = 7 * 24
    elif forecast_choice == "Next 7 Weeks":
        horizon = 7 * 7 * 24
    else:
        horizon = 30 * 24

    st.sidebar.write(f"📌 Forecast Hours: **{horizon}**")

    if st.sidebar.button("🔮 Generate AQI Forecast"):
        st.subheader(f"📈 AQI Forecast for {forecast_choice}")

        tabs = st.tabs(list(air_models.keys()))

        for tab, (name, model) in zip(tabs, air_models.items()):
            with tab:
                st.write(f"### Model: {name}")
                pred_df = forecast_future(model, horizon)

                st.dataframe(pred_df.tail(20))

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(pred_df["date"], pred_df["aqi_clean"])
                ax.set_title(f"{name} — AQI Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("AQI")
                ax.grid()
                st.pyplot(fig)

        st.success("✔ AQI Forecast Completed!")


# ============================================================
# ---------------- TAB 2: WATER QUALITY PREDICTION -----------
# ============================================================
# ============================================================
# ---------------- TAB 2: WATER QUALITY PREDICTION -----------
# ============================================================
with tab2:
    st.header("💧 Water Contamination Prediction Dashboard (20-Feature Model)")

    @st.cache_resource
    def load_water_models():
        model_paths = {
            "KNN": "KNN.joblib",
            "DecisionTree": "DecisionTree.joblib",
            "RandomForest": "RandomForest.joblib",
            "GradientBoosting": "GradientBoosting.joblib"
        }

        models = {}
        for name, path in model_paths.items():
            try:
                models[name] = joblib.load(path)
            except:
                st.warning(f"⚠ Missing model file: {path}")

        scaler = joblib.load("water_scaler.joblib")
        return models, scaler

    water_models, water_scaler = load_water_models()

    expected_cols = list(water_scaler.feature_names_in_)

    # Example / Sample values
    example_sample = {
        "aluminium": 0.5,
        "ammonia": 0.1,
        "arsenic": 0.002,
        "barium": 1.2,
        "cadmium": 0.0005,
        "chloramine": 2.5,
        "chromium": 0.01,
        "copper": 0.5,
        "flouride": 0.8,
        "bacteria": 0.0,
        "viruses": 0.0,
        "lead": 0.003,
        "nitrates": 5.0,
        "nitrites": 0.5,
        "mercury": 0.0002,
        "perchlorate": 4.0,
        "radium": 1.0,
        "selenium": 0.01,
        "silver": 0.05,
        "uranium": 1.0
    }

    st.sidebar.header("🧪 Enter Water Sample Values")

    # Load sample button
    if st.sidebar.button("🧪 Load Example Sample"):
        st.session_state.water_inputs = example_sample.copy()
    else:
        if "water_inputs" not in st.session_state:
            st.session_state.water_inputs = example_sample.copy()

    # Group inputs into categories
    metal_elements = [
        "aluminium","arsenic","barium","cadmium","chromium",
        "copper","lead","mercury","selenium","silver","uranium"
    ]
    chemical_contaminants = [
        "ammonia","chloramine","flouride","nitrates","nitrites","perchlorate"
    ]
    biological = ["bacteria","viruses"]
    radiological = ["radium"]

    def water_input():
        inputs = {}

        st.subheader("🧲 Metals & Heavy Elements")
        for col in metal_elements:
            inputs[col] = st.number_input(col, min_value=0.0, max_value=9999.0,
                                          value=float(st.session_state.water_inputs[col]))

        st.subheader("🧪 Chemical Contaminants")
        for col in chemical_contaminants:
            inputs[col] = st.number_input(col, min_value=0.0, max_value=9999.0,
                                          value=float(st.session_state.water_inputs[col]))

        st.subheader("🧬 Biological Contaminants")
        for col in biological:
            inputs[col] = st.number_input(col, min_value=0.0, max_value=1000.0,
                                          value=float(st.session_state.water_inputs[col]))

        st.subheader("☢ Radiological Contaminants")
        for col in radiological:
            inputs[col] = st.number_input(col, min_value=0.0, max_value=9999.0,
                                          value=float(st.session_state.water_inputs[col]))

        return pd.DataFrame([inputs])

    # Get input from UI
    water_df = water_input()

    # Fix column order to match scaler
    water_df = water_df[expected_cols]

    try:
        water_scaled = water_scaler.transform(water_df)
    except Exception as e:
        st.error("Scaler transform failed! Column mismatch. Please verify input feature names.")
        st.stop()

    st.subheader("🔍 Water Safety Prediction")

    if st.button("Predict Water Safety"):
        results = {}

        for name, model in water_models.items():
            pred = model.predict(water_scaled)[0]
            prob = model.predict_proba(water_scaled)[0][1]
            results[name] = {"Prediction": int(pred), "Probability(Safe)": prob}

        results_df = pd.DataFrame(results).T
        st.write(results_df)

        best_model = results_df["Probability(Safe)"].idxmax()

        st.success(f"🏆 Best Model Suggests: **{best_model}**")

        if results_df.loc[best_model, "Prediction"] == 1:
            st.markdown("### ✔ Water is **SAFE** to drink")
        else:
            st.markdown("### ❌ Water is **NOT SAFE** to drink")

# FOOTER
st.write("---")
st.write("Built with ❤️ using Streamlit and Machine Learning")
