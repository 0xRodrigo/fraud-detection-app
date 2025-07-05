import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime as dt
import requests
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

##Solo demo
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ─── Configuración y constantes ─────────────────────────────────────────
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS_DIR = Path("models")
base_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Inicializa el cliente de HuggingFace solo una vez
try:
    hf_client = InferenceClient(
        provider="auto",
        api_key=HF_TOKEN,
    )
except Exception as e:
    st.error(f"No se pudo inicializar el cliente de HuggingFace: {e}")
    st.stop()

# Cargar artefactos de estadística realista
def _safe_load(path, default=None):
    try:
        return joblib.load(path) if path.exists() else default
    except Exception as e:
        st.warning(f"Error cargando {path}: {e}")
        return default

# Intentar cargar estadísticas sintéticas reales
synth_stats_data = _safe_load(MODELS_DIR / "synth_stats.pkl", default=None)

# Perfiles demo como backup
SYNTHETIC_PROFILES = {
    "legitimate": {
        "V1": {"mean": -0.5, "std": 1.2},  "V2": {"mean": 0.2, "std": 1.1},  "V3": {"mean": 0.1, "std": 0.9},
        "V4": {"mean": 0.3, "std": 1.0},   "V5": {"mean": -0.1, "std": 0.8}, "V6": {"mean": 0.0, "std": 0.7},
        "V7": {"mean": 0.1, "std": 0.9},   "V8": {"mean": 0.0, "std": 0.8},  "V9": {"mean": -0.2, "std": 1.1},
        "V10": {"mean": 0.1, "std": 0.9},  "V11": {"mean": 0.2, "std": 1.0}, "V12": {"mean": -0.1, "std": 0.8},
        "V13": {"mean": 0.0, "std": 0.7},  "V14": {"mean": -0.3, "std": 1.2}, "V15": {"mean": 0.1, "std": 0.6},
        "V16": {"mean": -0.2, "std": 0.9}, "V17": {"mean": -0.1, "std": 0.8}, "V18": {"mean": 0.0, "std": 0.7},
        "V19": {"mean": 0.1, "std": 0.6},  "V20": {"mean": 0.0, "std": 0.5},  "V21": {"mean": 0.1, "std": 0.4},
        "V22": {"mean": 0.0, "std": 0.6},  "V23": {"mean": 0.0, "std": 0.3},  "V24": {"mean": 0.1, "std": 0.4},
        "V25": {"mean": 0.0, "std": 0.5},  "V26": {"mean": 0.1, "std": 0.4},  "V27": {"mean": 0.0, "std": 0.3},
        "V28": {"mean": 0.0, "std": 0.2},  "Amount": {"mean": 88.3, "std": 250.1}
    },
    "fraudulent": {
        "V1": {"mean": -2.8, "std": 2.1},  "V2": {"mean": 1.9, "std": 1.8},   "V3": {"mean": -1.2, "std": 1.5},
        "V4": {"mean": 2.8, "std": 2.2},   "V5": {"mean": -0.8, "std": 1.4},  "V6": {"mean": -1.4, "std": 1.3},
        "V7": {"mean": -0.9, "std": 1.2},  "V8": {"mean": 0.2, "std": 1.1},   "V9": {"mean": -0.5, "std": 1.0},
        "V10": {"mean": -1.8, "std": 1.9}, "V11": {"mean": 1.1, "std": 1.3},  "V12": {"mean": -1.5, "std": 1.6},
        "V13": {"mean": -0.3, "std": 0.8}, "V14": {"mean": -1.9, "std": 1.8}, "V15": {"mean": 0.4, "std": 0.9},
        "V16": {"mean": -0.8, "std": 1.2}, "V17": {"mean": -0.7, "std": 1.1}, "V18": {"mean": -0.2, "std": 0.8},
        "V19": {"mean": 0.3, "std": 0.7},  "V20": {"mean": 0.1, "std": 0.6},  "V21": {"mean": 0.2, "std": 0.5},
        "V22": {"mean": 0.5, "std": 0.8},  "V23": {"mean": -0.1, "std": 0.4}, "V24": {"mean": 0.2, "std": 0.5},
        "V25": {"mean": 0.4, "std": 0.6},  "V26": {"mean": 0.1, "std": 0.5},  "V27": {"mean": 0.0, "std": 0.4},
        "V28": {"mean": 0.0, "std": 0.3},  "Amount": {"mean": 122.2, "std": 256.7}
    }
}

def generate_synthetic_features(is_fraud: bool, amount: float, n_samples: int = 1) -> pd.DataFrame:
    """
    Genera un DataFrame de características sintéticas (V1-V28 + Amount) para una transacción,
    usando synth_stats.pkl si está disponible. Si no, perfiles demo.
    """
    cls = 1 if is_fraud else 0
    features = base_features
    if (
        synth_stats_data 
        and isinstance(synth_stats_data, dict)
        and cls in synth_stats_data
        and "mean" in synth_stats_data[cls] 
        and "cov" in synth_stats_data[cls]
    ):
        mean_vec = synth_stats_data[cls]["mean"]
        cov_mat  = synth_stats_data[cls]["cov"]
        X_synth = np.random.multivariate_normal(mean_vec, cov_mat, size=n_samples)
        df_synth = pd.DataFrame(X_synth, columns=features)
        if amount > 0:
            df_synth["Amount"] = amount
        return df_synth[features]
    # Fallback DEMO
    profile = SYNTHETIC_PROFILES["fraudulent" if is_fraud else "legitimate"]
    data = []
    for _ in range(n_samples):
        row = {f: np.random.normal(profile[f]["mean"], profile[f]["std"]) if f != "Amount" else amount for f in features}
        data.append(row)
    return pd.DataFrame(data, columns=features)

def _load_or_create_model(file_model, file_scaler):
    model = _safe_load(file_model)
    scaler = _safe_load(file_scaler)
    if model and scaler:
        return model, scaler
    # Mock fallback
    st.info("🔧 Creando modelos sintéticos para la demostración...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    n_samples = 1000
    X_leg = generate_synthetic_features(False, 100.0, n_samples//2)
    X_fra = generate_synthetic_features(True, 100.0, n_samples//2)
    X = pd.concat([X_leg, X_fra], ignore_index=True)
    y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    scaler.fit(X[['Amount']])
    X_scaled = X.copy()
    X_scaled['Amount'] = scaler.transform(X[['Amount']]).flatten()
    model.fit(X_scaled.values, y)
    st.success("✅ Modelos sintéticos creados")
    return model, scaler

def predict_with_model(model, scaler, features, event_df, threshold=0.5):
    X = event_df.reindex(columns=features, fill_value=0).copy()
    if 'Amount' in X.columns:
        X['Amount'] = scaler.transform(X[['Amount']]).flatten()
    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)
    return pred, prob

def get_card_brand(card_number: str):
    """
    Determina la marca de tarjeta y devuelve información relevante
    """
    card_number = card_number.replace(" ", "")
    
    if card_number.startswith("4"):
        return {
            "brand": "Visa",
            "logo": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Visa_Inc._logo.svg",
            "color": "linear-gradient(135deg, #1a1f71, #0066cc)"
        }
    elif card_number.startswith(("51", "52", "53", "54", "55")):
        return {
            "brand": "Mastercard",
            "logo": "https://upload.wikimedia.org/wikipedia/commons/2/2a/Mastercard-logo.svg",
            "color": "linear-gradient(135deg, #eb001b, #ff5f00)"
        }
    elif card_number.startswith(("34", "37")):
        return {
            "brand": "American Express",
            "logo": "https://upload.wikimedia.org/wikipedia/commons/f/fa/American_Express_logo_%282018%29.svg",
            "color": "linear-gradient(135deg, #006fcf, #00c3a7)"
        }
    else:
        return {
            "brand": "Genérica",
            "logo": "https://upload.wikimedia.org/wikipedia/commons/8/88/Credit-card-icon.png",
            "color": "linear-gradient(135deg, #4e54c8, #8f94fb)"
        }

def calculate_risk_score(prob_base, prob_optimized, amount):
    """
    Calcula un score de riesgo compuesto
    """
    # Factores de riesgo
    amount_risk = min(amount / 10000, 1.0)  # Normalizado hasta $10,000
    model_consensus = abs(prob_base - prob_optimized)
    
    # Score ponderado
    risk_score = (
        prob_optimized * 0.5 +
        amount_risk * 0.2 +
        model_consensus * 0.3
    )
    
    return min(risk_score, 1.0)

def format_card_number(card_number: str):
    """
    Formatea el número de tarjeta con espacios
    """
    clean_number = card_number.replace(" ", "")
    return " ".join([clean_number[i:i+4] for i in range(0, len(clean_number), 4)])

def obtener_interpretacion_ia(explanation_text, modo="SHAP"):
    """
    Llama a la API de Hugging Face para obtener una explicación sencilla del gráfico.
    """
    prompt = (
        "Eres un experto explicando interpretabilidad de modelos de fraude de tarjetas de crédito. "
        "Debes comparar las explicaciones de dos modelos. "
        "El usuario ve el siguiente resumen del gráfico de importancia de características usando " + modo + ":\n"
        f"{explanation_text}\n"
        "Redacta una explicación CLARA, en español neutro, para un usuario NO técnico. "
        "Usa frases cortas y ejemplos sencillos. Si una característica aparece en ambos modelos, resáltalo. "
        "Si hay diferencias claras, explícalas. Termina con una recomendación fácil de entender para el usuario."
    )
    try:
        completion = hf_client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error consultando la API de Hugging Face: {e}"

# ─── UI APP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Detector de Fraudes Avanzado", page_icon="💳", layout="wide")
base_model, base_scaler = _load_or_create_model(MODELS_DIR / "best_base_model.pkl", MODELS_DIR / "amount_scaler.pkl")
strategy_model, strategy_scaler = _load_or_create_model(MODELS_DIR / "best_strategy_model.pkl", MODELS_DIR / "strategy_scaler.pkl")

st.title("💳 Detector de Fraudes Avanzado")
st.markdown("### Sistema de análisis en tiempo real para transacciones con tarjeta de crédito")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.subheader("📦 Artefactos Cargados")

    base_status = "✅" if base_model else "❌"
    strat_status = "✅" if strategy_model else "❌"
    synth_status = "🧬 synth_stats.pkl" if synth_stats_data else "⚠️ Fallback (perfiles demo)"

    st.markdown(f"- **Modelo Base:** {base_status}")
    st.markdown(f"- **Modelo Optimizado:** {strat_status}")
    st.markdown(f"- **Scaler Base:** {'✅' if base_scaler else '❌'}")
    st.markdown(f"- **Scaler Optimizado:** {'✅' if strategy_scaler else '❌'}")
    st.markdown(f"- **Generación Sintética:** {synth_status}")

    if synth_stats_data:
        st.success("Generación sintética basada en estadísticas reales del dataset.")
    else:
        st.warning("Se emplean perfiles demo. Considera cargar 'synth_stats.pkl' para mayor realismo.")

     # ---- BOTÓN DE INFO Y MODAL ----
    youtube_url = "https://github.com/0xRodrigo/fraud-detection-app" 

    st.sidebar.markdown(
        f"""
        <style>
        .custom-button {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(90deg, #232e47 0%, #424d6e 100%);
            color: #fff;
            border-radius: 0.6em;
            padding: 0.7em 1.2em;
            font-weight: 600;
            font-size: 1.05em;
            margin-top: 16px;
            margin-bottom: 16px;
            text-decoration: none;
            box-shadow: 0 2px 12px 0 rgba(0,0,0,0.07);
            transition: background 0.2s;
            border: none;
        }}
        .custom-button:hover {{
            background: linear-gradient(90deg, #5161b3 0%, #4254a6 100%);
            color: #f8f8f8;
            text-decoration: none;
        }}
        .custom-button .icon {{
            margin-right: 0.6em;
            font-size: 1.2em;
        }}
        </style>
        <a class="custom-button" href="{youtube_url}" target="_blank">
            <span class="icon">ℹ️</span> Descripcion del aplicativo (nueva pestaña)
        </a>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader("⚙️ Parámetros de Simulación")
    strategy_thresh = st.slider("Umbral modelo optimizado", 0.01, 0.99, 0.30, 0.01)
    fraud_probability = st.slider("Probabilidad de fraude (%)", 0, 100, 50)
    force_fraud = st.checkbox("Forzar fraude", value=False)

    # Estadísticas de sesión
    if 'transactions_processed' not in st.session_state:
        st.session_state.transactions_processed = 0
        st.session_state.frauds_detected = 0
    
    st.subheader("📊 Estadísticas de Sesión")
    st.metric("Transacciones procesadas", st.session_state.transactions_processed)
    st.metric("Fraudes detectados", st.session_state.frauds_detected)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💳 Datos de la Tarjeta")
    with st.form(key="payment_form"):
        card_name = st.text_input("👤 Nombre del titular", placeholder="Juan Pérez")
        card_number = st.text_input("💳 Número de tarjeta", placeholder="4111 1111 1111 1111", max_chars=19)
        col3, col4 = st.columns(2)
        with col3:
            exp_date = st.text_input("📅 Expiración (MM/AA)", placeholder="12/25", max_chars=5)
        with col4:
            cvv = st.text_input("🔐 CVV", placeholder="123", max_chars=3, type="password")
        amount = st.number_input(
            "💵 Monto ($)",
            min_value=0.01, max_value=100_000.0, step=0.01, value=100.00, format="%.2f"
        )
        merchant_category = st.selectbox(
            "🏪 Categoría del comerciante",
            ["Supermercado", "Gasolinera", "Restaurante", "Tienda en línea", "Cajero automático", "Otro"]
        )
        location = st.selectbox(
            "📍 Ubicación",
            ["Quito", "Guayaquil", "Cuenca", "Internacional", "Otra"]
        )
        submitted = st.form_submit_button("🔍 ANALIZAR TRANSACCIÓN", use_container_width=True)

with col2:
    st.subheader("💳 Vista Previa de la Tarjeta")
    
    # Obtener información de la tarjeta
    card_info = get_card_brand(card_number if card_number else "")
    formatted_number = format_card_number(card_number) if card_number else "**** **** **** ****"
    
    # Renderizar tarjeta
    st.markdown(f"""
    <div style="background: {card_info['color']};
                border-radius: 15px;
                padding: 25px;
                color: white;
                width: 350px;
                height: 200px;
                margin: auto;
                font-family: 'Courier New', monospace;
                box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                position: relative;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 14px; opacity: 0.8;">{card_info['brand']}</div>
            <img src="{card_info['logo']}" width="50" style="filter: brightness(0) invert(1);">
        </div>
        <div style="margin-top: 40px;">
            <div style="font-size: 20px; letter-spacing: 3px; margin-bottom: 15px;">
                {formatted_number}
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-size: 10px; opacity: 0.7;">TITULAR</div>
                    <div style="font-size: 14px;">{card_name.upper() if card_name else 'NOMBRE APELLIDO'}</div>
                </div>
                <div>
                    <div style="font-size: 10px; opacity: 0.7;">EXPIRA</div>
                    <div style="font-size: 14px;">{exp_date if exp_date else 'MM/AA'}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if submitted:
    # Limpiar explicaciones anteriores de IA
    st.session_state['explicacion_shap'] = ""
    st.session_state['explicacion_lime'] = ""
    st.session_state.transactions_processed += 1
    with st.spinner("🔍 Analizando transacción..."):
        progress_bar = st.progress(0)
        is_fraud = force_fraud or (np.random.rand() < fraud_probability / 100)
        event = generate_synthetic_features(is_fraud, amount, 1)
        progress_bar.progress(25)
        base_pred, base_prob = predict_with_model(base_model, base_scaler, base_features, event, 0.5)
        progress_bar.progress(50)
        strategy_pred, strategy_prob = predict_with_model(strategy_model, strategy_scaler, base_features, event, strategy_thresh)
        progress_bar.progress(75)
        risk_score = 0.5 * strategy_prob + 0.2 * min(amount/10000, 1.0) + 0.3 * abs(base_prob - strategy_prob)
        progress_bar.progress(100)
        if strategy_pred == 1:
            st.session_state.frauds_detected += 1
    st.success("✅ Análisis completado")

    st.session_state["event"] = event
    st.session_state["amount"] = amount
    st.session_state["merchant_category"] = merchant_category
    st.session_state["location"] = location
    st.session_state["base_pred"] = base_pred
    st.session_state["base_prob"] = base_prob
    st.session_state["strategy_pred"] = strategy_pred
    st.session_state["strategy_prob"] = strategy_prob
    st.session_state["risk_score"] = risk_score
    st.session_state["is_fraud"] = is_fraud
    st.session_state["card_info"] = card_info  

show_analysis = (
    ('event' in st.session_state) and
    ('base_pred' in st.session_state)
)

if show_analysis:
    # Recupera los datos guardados
    event = st.session_state["event"]
    amount = st.session_state["amount"]
    merchant_category = st.session_state["merchant_category"]
    location = st.session_state["location"]
    base_pred = st.session_state["base_pred"]
    base_prob = st.session_state["base_prob"]
    strategy_pred = st.session_state["strategy_pred"]
    strategy_prob = st.session_state["strategy_prob"]
    risk_score = st.session_state["risk_score"]
    is_fraud = st.session_state["is_fraud"]
    card_info = st.session_state["card_info"]

    # Resultados principales
    st.subheader("🎯 Resultados del Análisis")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Modelo Base",
            "FRAUDE" if base_pred else "LEGÍTIMA",
            f"{base_prob:.1%}",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Modelo Optimizado",
            "FRAUDE" if strategy_pred else "LEGÍTIMA",
            f"{strategy_prob:.1%}",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "Score de Riesgo",
            f"{risk_score:.1%}",
            "Alto" if risk_score > 0.7 else "Medio" if risk_score > 0.3 else "Bajo"
        )

    with col4:
        decision_color = "🔴" if strategy_pred else "🟢"
        st.metric(
            "Decisión Final",
            f"{decision_color} {'BLOQUEAR' if strategy_pred else 'APROBAR'}",
            f"Confianza: {max(strategy_prob, 1-strategy_prob):.1%}"
        )

    # Gráfico de probabilidades
    st.subheader("📊 Análisis Comparativo")

    fig = go.Figure()

    # Barras de probabilidad
    fig.add_trace(go.Bar(
        name="Modelo Base",
        x=["Legítima", "Fraude"],
        y=[1-base_prob, base_prob],
        marker_color="lightblue"
    ))

    fig.add_trace(go.Bar(
        name="Modelo Optimizado",
        x=["Legítima", "Fraude"],
        y=[1-strategy_prob, strategy_prob],
        marker_color="salmon"
    ))

    # Líneas de umbral
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                    annotation_text="Umbral Base (50%)")
    fig.add_hline(y=strategy_thresh, line_dash="dot", line_color="red",
                    annotation_text=f"Umbral Optimizado ({strategy_thresh:.0%})")

    fig.update_layout(
        title="Probabilidades de Clasificación",
        yaxis_title="Probabilidad",
        barmode="group",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretabilidad
    st.subheader("🔍 Interpretación del Modelo")

    try:
        tab_shap, tab_lime = st.tabs(["📊 SHAP (Importancia de Características)", "🎯 LIME (Explicación Local)"])

        # DataFrame 2D, columnas alineadas
        event_df = event[base_features] if isinstance(event, pd.DataFrame) else pd.DataFrame([event], columns=base_features)

        # ========== SHAP ==========
        with tab_shap:
            col_base, col_opt = st.columns(2)

            # --- SHAP MODELO BASE ---
            with col_base:
                st.markdown("##### Modelo Base")
                explainer_base = shap.TreeExplainer(base_model)
                shap_vals_base = explainer_base.shap_values(event_df)
                if isinstance(shap_vals_base, list) and len(shap_vals_base) == 2:
                    vals_base = shap_vals_base[1][0]
                else:
                    vals_base = shap_vals_base[0] if hasattr(shap_vals_base, "__len__") and len(shap_vals_base) else shap_vals_base

                feature_importance_base_full = pd.DataFrame({
                    'feature': base_features,
                    'shap_value': np.array(vals_base).flatten()[:len(base_features)]
                }).sort_values('shap_value', key=abs, ascending=False).head(10)

                fig_base = px.bar(
                    feature_importance_base_full,
                    x='shap_value', y='feature', orientation='h',
                    title='Top 10 (Base)', color='shap_value', color_continuous_scale='RdYlBu'
                )
                st.plotly_chart(fig_base, use_container_width=True)

            # --- SHAP MODELO OPTIMIZADO ---
            with col_opt:
                st.markdown("##### Modelo Optimizado")
                explainer_opt = shap.TreeExplainer(strategy_model)
                shap_vals_opt = explainer_opt.shap_values(event_df)
                if isinstance(shap_vals_opt, list) and len(shap_vals_opt) == 2:
                    vals_opt = shap_vals_opt[1][0]
                else:
                    vals_opt = shap_vals_opt[0] if hasattr(shap_vals_opt, "__len__") and len(shap_vals_opt) else shap_vals_opt

                feature_importance_opt_full = pd.DataFrame({
                    'feature': base_features,
                    'shap_value': np.array(vals_opt).flatten()[:len(base_features)]
                }).sort_values('shap_value', key=abs, ascending=False).head(10)

                fig_opt = px.bar(
                    feature_importance_opt_full,
                    x='shap_value', y='feature', orientation='h',
                    title='Top 10 (Optimizado)', color='shap_value', color_continuous_scale='RdYlBu'
                )
                st.plotly_chart(fig_opt, use_container_width=True)

            # --- topN para la explicación IA (no afecta gráficos)
            topN = st.radio("¿Cuántas características comparar con IA?", [5, 10], index=0, horizontal=True)

            feature_importance_base_ia = feature_importance_base_full.head(topN)
            feature_importance_opt_ia  = feature_importance_opt_full.head(topN)

                # --- INTERPRETACIÓN IA SHAP ---

            if 'explicacion_shap' not in st.session_state:
                st.session_state['explicacion_shap'] = ""

            descripcion_comparativa_shap = (
                f"Análisis de interpretabilidad SHAP para una transacción de ${amount:,.2f}.\n\n"
                f"Resultado del Modelo Base: {'FRAUDE' if base_pred else 'LEGÍTIMA'} (prob: {base_prob:.2%})\n"
                f"Resultado del Modelo Optimizado: {'FRAUDE' if strategy_pred else 'LEGÍTIMA'} (prob: {strategy_prob:.2%})\n\n"
                "Top características según SHAP:\n"
                "Modelo Base:\n" +
                "\n".join([f"{f['feature']}: {f['shap_value']:.2f}" for f in feature_importance_base_ia.to_dict(orient='records')]) +
                "\n\nModelo Optimizado:\n" +
                "\n".join([f"{f['feature']}: {f['shap_value']:.2f}" for f in feature_importance_opt_ia.to_dict(orient='records')])
            )
            with st.expander("💡 Interpretación comparativa SHAP con IA"):
                if st.button("Interpretar ahora", key="btn_shap"):
                    with st.spinner("Consultando IA..."):
                        st.session_state['explicacion_shap'] = obtener_interpretacion_ia(descripcion_comparativa_shap, modo="SHAP")
                if st.session_state['explicacion_shap']:
                    st.info(st.session_state['explicacion_shap'])

        # ========== LIME ==========
        with tab_lime:
            col_base, col_opt = st.columns(2)
            # Datos sintéticos de entrenamiento
            training_leg = generate_synthetic_features(False, float(amount), 50)
            training_fraud = generate_synthetic_features(True, float(amount), 50)
            training_data = pd.concat([training_leg, training_fraud], ignore_index=True)
            training_data = training_data[base_features]

            # Función de predicción LIME genérica
            def lime_predict_fn(X, model, scaler):
                X_df = pd.DataFrame(X, columns=base_features)
                if 'Amount' in X_df.columns:
                    X_df['Amount'] = scaler.transform(X_df[['Amount']]).flatten()
                return model.predict_proba(X_df.values)

            instance = event_df.iloc[0].to_numpy().flatten()[:len(base_features)]

            # --- LIME MODELO BASE ---
            with col_base:
                st.markdown("##### Modelo Base")
                explainer_lime_base = LimeTabularExplainer(
                    training_data=training_data.values,
                    feature_names=base_features,
                    class_names=['Legítima', 'Fraude'],
                    mode='classification'
                )
                explanation_base = explainer_lime_base.explain_instance(
                    instance,
                    lambda X: lime_predict_fn(X, base_model, base_scaler),
                    num_features=10
                )
                lime_df_base = pd.DataFrame(explanation_base.as_list(), columns=['feature', 'importance'])
                lime_df_base = lime_df_base.sort_values('importance', key=abs, ascending=False)
                fig_lime_base = px.bar(
                    lime_df_base,
                    x='importance', y='feature', orientation='h',
                    title='LIME Top 10 (Base)', color='importance', color_continuous_scale='RdYlBu'
                )
                st.plotly_chart(fig_lime_base, use_container_width=True)

            # --- LIME MODELO OPTIMIZADO ---
            with col_opt:
                st.markdown("##### Modelo Optimizado")
                explainer_lime_opt = LimeTabularExplainer(
                    training_data=training_data.values,
                    feature_names=base_features,
                    class_names=['Legítima', 'Fraude'],
                    mode='classification'
                )
                explanation_opt = explainer_lime_opt.explain_instance(
                    instance,
                    lambda X: lime_predict_fn(X, strategy_model, strategy_scaler),
                    num_features=10
                )
                lime_df_opt = pd.DataFrame(explanation_opt.as_list(), columns=['feature', 'importance'])
                lime_df_opt = lime_df_opt.sort_values('importance', key=abs, ascending=False)
                fig_lime_opt = px.bar(
                    lime_df_opt,
                    x='importance', y='feature', orientation='h',
                    title='LIME Top 10 (Optimizado)', color='importance', color_continuous_scale='RdYlBu'
                )
                st.plotly_chart(fig_lime_opt, use_container_width=True)

            # # === Interpretación comparativa LIME con IA ===

            lime_df_base = lime_df_base.head(topN)
            lime_df_opt  = lime_df_opt.head(topN)

            if 'explicacion_lime' not in st.session_state:
                st.session_state['explicacion_lime'] = ""
            descripcion_comparativa_lime = (
                f"Análisis de interpretabilidad LIME para una transacción de ${amount:,.2f}.\n\n"
                f"Resultado del Modelo Base: {'FRAUDE' if base_pred else 'LEGÍTIMA'} (prob: {base_prob:.2%})\n"
                f"Resultado del Modelo Optimizado: {'FRAUDE' if strategy_pred else 'LEGÍTIMA'} (prob: {strategy_prob:.2%})\n\n"
                "Top características según LIME:\n"
                "Modelo Base:\n" +
                "\n".join([f"{f['feature']}: {f['importance']:.2f}" for f in lime_df_base.to_dict(orient='records')]) +
                "\n\nModelo Optimizado:\n" +
                "\n".join([f"{f['feature']}: {f['importance']:.2f}" for f in lime_df_opt.to_dict(orient='records')])
            )
            with st.expander("💡 Interpretación comparativa LIME con IA"):
                if st.button("Interpretar ahora", key="btn_lime"):
                    with st.spinner("Consultando IA..."):
                        st.session_state['explicacion_lime'] = obtener_interpretacion_ia(descripcion_comparativa_lime, modo="LIME")
                if st.session_state['explicacion_lime']:
                    st.info(st.session_state['explicacion_lime'])

    except Exception as e:
        st.warning(f"Error en interpretabilidad: {e}")
        st.info("La interpretabilidad requiere modelos entrenados y datos válidos. Verifica que los artefactos estén alineados y actualizados.")
        
    # Información adicional
    with st.expander("🔍 Detalles Técnicos"):
        st.write("**Información de la Transacción:**")
        st.write(f"- Monto: ${amount:,.2f}")
        st.write(f"- Categoría: {merchant_category}")
        st.write(f"- Ubicación: {location}")
        st.write(f"- Marca de tarjeta: {card_info['brand']}")
        st.write(f"- Clase real (simulada): {'Fraude' if is_fraud else 'Legítima'}")
        
        st.write("**Características Sintéticas Generadas:**")
        st.dataframe(event.round(4))

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    © {dt.date.today().year} · Sistema de Detección de Fraudes · Demostración con datos sintéticos · Uso educativo
    <br><br>
    <span style="display: inline-flex; align-items: center; gap: 22px; justify-content: center;">
        <span>
            <a href="www.linkedin.com/in/rodrigo-ullauri" target="_blank" style="text-decoration: none;">
                <img src="https://media.licdn.com/dms/image/v2/D4E03AQGCVUD3mSOhhQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1722727977537?e=1756944000&v=beta&t=nAH7MvB4bABSurVq748El2xW5KCizOq_bs3syJosVYc" alt="Rodrigo" style="width:80px; height:80px; border-radius:70%; border: 2.5px solid #0a66c2;">
                <br>
                <span style="color: #0a66c2; font-size:0.9em;">Rodrigo</span>
            </a>
        </span>
        <span>
            <a href="https://www.linkedin.com/in/brigith-ullauri/" target="_blank" style="text-decoration: none;">
                <img src="https://media.licdn.com/dms/image/v2/C5603AQFGvdD51MIIdw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1643931954513?e=1756944000&v=beta&t=ebuqqKoVJgWSnRIgARzODq8Mdu401QocD-9vWo-dZI8" alt="Brigith" style="width:80px; height:80px; border-radius:70%; border: 2.5px solid #0a66c2;">
                <br>
                <span style="color: #0a66c2; font-size:0.9em;">Brigith</span>
            </a>
        </span>
        <span>
            <a href="https://www.linkedin.com/in/joffre-velasco-53ba9a48/" target="_blank" style="text-decoration: none;">
                <img src="https://media.licdn.com/dms/image/v2/D4E03AQGoF77TCOkWaA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1720092274171?e=1756944000&v=beta&t=hYKz87-Bs9WpukTRRFft5IW1Dena6Qviqd4zUZJcBeU" alt="Joffre" style="width:80px; height:80px; border-radius:70%; border: 2.5px solid #0a66c2;">
                <br>
                <span style="color: #0a66c2; font-size:0.9em;">Joffre</span>
            </a>
        </span>
        <span>
            <a href="https://www.linkedin.com/in/darwin-ipiales-0356ba89/" target="_blank" style="text-decoration: none;">
                <img src="https://media.licdn.com/dms/image/v2/D4D35AQH5_9Xd2Z9EDA/profile-framedphoto-shrink_400_400/profile-framedphoto-shrink_400_400/0/1736982702164?e=1752357600&v=beta&t=LRpIaDdvJhbVNWMMHy3mq_6FjhIfQk8rinzroQRxE28" alt="Darwin" style="width:80px; height:80px; border-radius:70%; border: 2.5px solid #0a66c2;">
                <br>
                <span style="color: #0a66c2; font-size:0.9em;">Darwin</span>
            </a>
        </span>
    </span>
    <br>
    <span style="font-size: 0.9em; color: #888;">¿Dudas técnicas? ¡Contáctanos por LinkedIn!</span>
</div>
""", unsafe_allow_html=True)
