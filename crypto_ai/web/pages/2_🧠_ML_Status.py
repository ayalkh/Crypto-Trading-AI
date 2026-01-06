import streamlit as st
import pandas as pd
import os
import glob
import joblib

st.set_page_config(page_title="ML Status", page_icon="🧠", layout="wide")

st.title("🧠 ML Model Status")

models_dir = 'ml_models'

if not os.path.exists(models_dir):
    st.error(f"Models directory '{models_dir}' not found.")
    st.stop()

# Helper to scan models
def get_model_inventory():
    files = glob.glob(os.path.join(models_dir, "*.joblib"))
    models = []
    for f in files:
        if 'scaler' in f or 'features' in f:
            continue
        models.append(os.path.basename(f))
    return models

models = get_model_inventory()

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Models", len(models))
    
with col2:
    st.metric("Model Health", "Good" if len(models) > 0 else "Needs Training")

st.subheader("Model Inventory")

if models:
    df_models = pd.DataFrame(models, columns=["Filename"])
    # Attempt to parse info from filename
    # Format: {symbol}_{timeframe}_{type}_{algo}.joblib
    # Example: BTC_USDT_1h_price_xgboost.joblib
    
    def parse_filename(name):
        parts = name.replace('.joblib', '').split('_')
        if len(parts) >= 4:
            # Reconstruct symbol (BTC_USDT -> BTC/USDT)
            symbol = f"{parts[0]}/{parts[1]}"
            timeframe = parts[2]
            pred_type = parts[3]
            algo = parts[4] if len(parts) > 4 else "Unknown"
            return pd.Series([symbol, timeframe, pred_type, algo])
        return pd.Series(["-", "-", "-", "-"])

    # Temporarily suppress pandas warnings if needed or import pandas
    import pandas as pd
    meta = df_models['Filename'].apply(parse_filename)
    meta.columns = ['Symbol', 'Timeframe', 'Type', 'Algo']
    
    st.dataframe(pd.concat([df_models, meta], axis=1), use_container_width=True)
else:
    st.info("No trained models found.")

st.markdown("---")
st.subheader("Training Actions")

col_train1, col_train2 = st.columns(2)
with col_train1:
    st.info("Training is currently only available via CLI.")
    st.code("python crypto_control_center.py")

