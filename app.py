import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from GradientDescent import GradientDescent

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Diabetes Gradient Descent Studio",
    page_icon="📉",
    layout="wide"
)

# -----------------------------------------------------
# DESIGN TOKENS
# Palette: near-black navy base, amber = "signal" (glucose/warmth),
# teal = "cost descending / cooling" — tied to the subject, not a default.
# -----------------------------------------------------
BG = "#0A0E17"
SURFACE = "#121826"
BORDER = "#232B3D"
INK = "#EDF0F5"
MUTED = "#8A93A8"
AMBER = "#F2A65A"
TEAL = "#45C7B5"

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: {FONT_BODY};
    }}
    .stApp {{
        background-color: {BG};
        color: {INK};
    }}
    h1, h2, h3, h4 {{
        font-family: {FONT_DISPLAY} !important;
        letter-spacing: -0.01em;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: {FONT_MONO};
        color: {AMBER};
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}
    .studio-chip {{
        display: inline-block;
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 999px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-family: {FONT_MONO};
        font-size: 0.82rem;
        color: {TEAL};
    }}
    .studio-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 14px;
    }}
    .studio-eyebrow {{
        font-family: {FONT_MONO};
        color: {AMBER};
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .studio-hero-number {{
        font-family: {FONT_DISPLAY};
        font-size: 3.1rem;
        font-weight: 700;
        color: {INK};
        line-height: 1.1;
    }}
    hr {{
        border-color: {BORDER} !important;
    }}
</style>
""", unsafe_allow_html=True)


def themed_fig(fig, height=440):
    """Apply the studio's dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_BODY, color=INK, size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    return diabetes, df

diabetes, df = load_data()

X = df.drop(columns=["bp"])
y = df["bp"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.markdown(f"""
<div style="padding: 6px 0 18px 0;">
    <div class="studio-eyebrow">ML from Scratch</div>
    <div style="font-family:{FONT_DISPLAY}; font-size:1.3rem; font-weight:700; color:{INK};">
        Gradient Descent Studio
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset", "Preprocessing", "Models", "Cost Landscape", "Conclusion"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span style='color:{MUTED}; font-size:0.85rem;'>💡 Adjust sliders on "
    f"<b style='color:{TEAL}'>Models</b> and watch the descent path change on "
    f"<b style='color:{AMBER}'>Cost Landscape</b>.</span>",
    unsafe_allow_html=True
)

# =====================================================
# HOME
# =====================================================
if page == "Home":
    st.markdown("<div class='studio-eyebrow'>Diabetes · Blood Pressure · Regression</div>", unsafe_allow_html=True)
    st.markdown("<div class='studio-hero-number'>Predicting Blood Pressure</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED}; font-size:1.05rem; max-width:640px;'>"
        f"A from-scratch Gradient Descent engine, benchmarked against Scikit-learn's SGD Regressor, "
        f"on the classic sklearn Diabetes dataset.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Target", "Blood Pressure")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='studio-eyebrow'>Feature set</div>", unsafe_allow_html=True)
    chips = "".join([f"<span class='studio-chip'>{c}</span>" for c in df.columns])
    st.markdown(chips, unsafe_allow_html=True)

# =====================================================
# DATASET
# =====================================================
elif page == "Dataset":
    st.markdown("<div class='studio-eyebrow'>Exploration</div>", unsafe_allow_html=True)
    st.markdown("## Dataset & Correlation")

    tab1, tab2 = st.tabs(["Preview", "Statistical Summary"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Feature correlation with Blood Pressure")

    corr = df.corr(numeric_only=True)["bp"].drop("bp").sort_values()
    colors = [TEAL if v < 0 else AMBER for v in corr.values]

    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=colors))
    fig.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="")
    st.plotly_chart(themed_fig(fig, height=420), use_container_width=True)

# =====================================================
# PREPROCESSING
# =====================================================
elif page == "Preprocessing":
    st.markdown("<div class='studio-eyebrow'>Preparation</div>", unsafe_allow_html=True)
    st.markdown("## Feature Scaling")

    c1, c2 = st.columns(2)
    c1.metric("Training Samples", len(X_train))
    c2.metric("Testing Samples", len(X_test))

    st.markdown("<br>", unsafe_allow_html=True)
    feature = st.selectbox("Feature to inspect", X.columns)
    idx = X.columns.get_loc(feature)

    before = X_train.iloc[:, idx]
    after = X_train_scaled[:, idx]

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure(go.Histogram(x=before, marker_color=AMBER, nbinsx=20))
        fig1.update_layout(title=f"Original — {feature}")
        st.plotly_chart(themed_fig(fig1, height=360), use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Histogram(x=after, marker_color=TEAL, nbinsx=20))
        fig2.update_layout(title=f"Scaled — {feature}")
        st.plotly_chart(themed_fig(fig2, height=360), use_container_width=True)

# =====================================================
# MODELS
# =====================================================
elif page == "Models":
    st.markdown("<div class='studio-eyebrow'>Optimization</div>", unsafe_allow_html=True)
    st.markdown("## Model Workspace")

    tab_sgd, tab_gd = st.tabs(["Scikit-learn SGD Regressor", "Custom Gradient Descent"])

    # ---------------- SGD Regressor ----------------
    with tab_sgd:
        alpha = st.slider("Regularization Alpha", 0.000001, 1.0, 0.01, format="%.6f", key="alpha")

        model = SGDRegressor(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("MSE", f"{mse:.3f}")
        c3.metric("R² Score", f"{r2:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}).sort_values("Coefficient")
            fig = go.Figure(go.Bar(
                x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
                marker_color=[TEAL if v < 0 else AMBER for v in coef_df["Coefficient"]]
            ))
            fig.update_layout(title="Feature Coefficients")
            st.plotly_chart(themed_fig(fig, height=400), use_container_width=True)

        with col_right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=pred, mode="markers",
                                      marker=dict(color=AMBER, size=7, opacity=0.75), name="Prediction"))
            mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                      line=dict(color=MUTED, dash="dash"), name="Ideal"))
            fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual BP", yaxis_title="Predicted BP")
            st.plotly_chart(themed_fig(fig, height=400), use_container_width=True)

    # ---------------- Custom Gradient Descent ----------------
    with tab_gd:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.1, key="gd_lr")
        with col_s2:
            iterations = st.slider("Iterations", 10, 500, 100, key="gd_iter")

        bmi_idx = X.columns.get_loc("bmi")
        X_train_new = X_train_scaled[:, [bmi_idx]]
        X_test_new = X_test_scaled[:, [bmi_idx]]

        gd = GradientDescent(alfa=learning_rate, iterations=iterations)
        gd.fit(X_train_new, y_train.values, track_history=True)

        slope = gd.m
        intercept = gd.c
        history = np.array([[h_m[0], h_c, h_cost] for h_m, h_c, h_cost in gd.history])

        pred = (X_test_new @ slope + intercept).ravel()
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slope", f"{float(slope[0]):.3f}")
        c2.metric("Intercept", f"{intercept:.3f}")
        c3.metric("MSE", f"{mse:.3f}")
        c4.metric("R² Score", f"{r2:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig = go.Figure(go.Scatter(y=history[:, 2], mode="lines", line=dict(color=AMBER, width=3)))
            fig.update_layout(title="Cost vs Iteration", xaxis_title="Iteration", yaxis_title="Cost (MSE)")
            st.plotly_chart(themed_fig(fig, height=380), use_container_width=True)

        with col_g2:
            line_x = np.linspace(X_train_new[:, 0].min(), X_train_new[:, 0].max(), 100)
            line_y = line_x * float(slope[0]) + intercept

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_train_new[:, 0], y=y_train, mode="markers",
                                      marker=dict(color=TEAL, size=6, opacity=0.5), name="Data"))
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                      line=dict(color=AMBER, width=3), name="Fit"))
            fig.update_layout(title="Fitted Line — BMI vs BP", xaxis_title="BMI (scaled)", yaxis_title="Blood Pressure")
            st.plotly_chart(themed_fig(fig, height=380), use_container_width=True)

# =====================================================
# COST LANDSCAPE (3D)
# =====================================================
elif page == "Cost Landscape":
    st.markdown("<div class='studio-eyebrow'>The Descent</div>", unsafe_allow_html=True)
    st.markdown("## 3D Cost Surface & Optimization Path")
    st.markdown(
        f"<p style='color:{MUTED};'>Every point on this surface is a possible (slope, intercept) pair — "
        f"height is the resulting error. The line traces gradient descent's actual path down to the minimum. "
        f"Drag to rotate.</p>",
        unsafe_allow_html=True
    )

    bmi_idx = X.columns.get_loc("bmi")
    X_train_new = X_train_scaled[:, [bmi_idx]]

    gd = GradientDescent(alfa=0.1, iterations=100)
    gd.fit(X_train_new, y_train.values, track_history=True)

    slope = gd.m
    intercept = gd.c
    history = np.array([[h_m[0], h_c, h_cost] for h_m, h_c, h_cost in gd.history])

    def cost(m, c, x, y):
        pred = x * m + c
        return np.mean((pred.ravel() - y) ** 2)

    m_vals = np.linspace(float(slope[0]) - 10, float(slope[0]) + 10, 50)
    c_vals = np.linspace(intercept - 10, intercept + 10, 50)
    M, C = np.meshgrid(m_vals, c_vals)
    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = cost(M[i, j], C[i, j], X_train_new, y_train.values)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=M, y=C, z=Z,
        colorscale=[[0, SURFACE], [0.5, TEAL], [1, AMBER]],
        opacity=0.85, showscale=True,
        colorbar=dict(title="Cost", tickfont=dict(color=INK)),
        contours_z=dict(show=True, usecolormap=True, project_z=True)
    ))

    fig.add_trace(go.Scatter3d(
        x=history[:, 0], y=history[:, 1], z=history[:, 2],
        mode="lines+markers",
        line=dict(color=INK, width=4),
        marker=dict(size=3, color=np.arange(len(history)), colorscale=[[0, TEAL], [1, AMBER]]),
        name="Descent Path"
    ))

    fig.add_trace(go.Scatter3d(
        x=[float(slope[0])], y=[intercept], z=[float(history[-1, 2])],
        mode="markers", marker=dict(size=7, color=AMBER, symbol="diamond"),
        name="Converged"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Slope", backgroundcolor=SURFACE, gridcolor=BORDER, color=INK),
            yaxis=dict(title="Intercept", backgroundcolor=SURFACE, gridcolor=BORDER, color=INK),
            zaxis=dict(title="Cost", backgroundcolor=SURFACE, gridcolor=BORDER, color=INK),
            bgcolor=SURFACE,
        ),
        paper_bgcolor=SURFACE,
        font=dict(family=FONT_BODY, color=INK),
        margin=dict(l=0, r=0, t=10, b=0),
        height=640,
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CONCLUSION
# =====================================================
else:
    st.markdown("<div class='studio-eyebrow'>Summary</div>", unsafe_allow_html=True)
    st.markdown("## Architecture & Takeaways")

    cards = [
        ("Modular Design", "An isolated GradientDescent.py module implements the custom optimization loop, kept separate from the app/UI layer."),
        ("Tracked Convergence", "Every iteration's slope, intercept, and cost are recorded, powering both the cost curve and the 3D descent path."),
        ("Benchmarked, Not Assumed", "The scratch implementation is validated against Scikit-learn's SGDRegressor on the same data and split."),
    ]

    for title, desc in cards:
        st.markdown(f"""
        <div class="studio-card">
            <div style="font-family:{FONT_DISPLAY}; font-weight:600; font-size:1.05rem; color:{AMBER}; margin-bottom:6px;">{title}</div>
            <div style="color:{MUTED};">{desc}</div>
        </div>
        """, unsafe_allow_html=True)