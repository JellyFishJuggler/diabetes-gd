import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from GradientDescent import GradientDescent

# -----------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------
st.set_page_config(
    page_title="Diabetes Gradient Descent Studio",
    page_icon="📉",
    layout="wide"
)

# Custom CSS for modern card containers and clean typography
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset",
        "Preprocessing",
        "SGD Regressor",
        "Gradient Descent",
        "3D Cost Surface",
        "Conclusion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the custom sliders under *Gradient Descent* to inspect live learning updates.")

# =====================================================
# HOME
# =====================================================
if page == "Home":
    st.title("📉 Diabetes Gradient Descent Studio")
    st.markdown("An interactive workspace exploring **Linear Regression** via Scikit-learn and custom **Gradient Descent from Scratch**[cite: 1].")
    
    st.markdown("---")

    # Grid layout for high-level metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(df))
    c2.metric("Total Features", len(df.columns))
    c3.metric("Target Variable", "Blood Pressure")

    st.markdown("### 🧩 Feature Overview")
    st.write(list(df.columns))

# =====================================================
# DATASET
# =====================================================
elif page == "Dataset":
    st.title("📊 Dataset Overview & Correlation")

    # Side-by-side layout for preview and stats
    tab1, tab2 = st.tabs(["Data Preview", "Statistical Summary"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlation with Blood Pressure")
    
    corr = df.corr(numeric_only=True)["bp"].sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    corr.plot(kind="barh", ax=ax, color="#4f46e5")
    ax.set_xlabel("Correlation Coefficient")
    ax.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig)

# =====================================================
# PREPROCESSING
# =====================================================
elif page == "Preprocessing":
    st.title("⚙️ Feature Scaling Analysis")

    c1, c2 = st.columns(2)
    c1.metric("Training Samples", len(X_train))
    c2.metric("Testing Samples", len(X_test))

    st.markdown("---")

    feature = st.selectbox("Select Feature to Visualize Distribution", X.columns)
    idx = X.columns.get_loc(feature)

    before = X_train.iloc[:, idx]
    after = X_train_scaled[:, idx]

    # Grid Layout: Side-by-side distributions
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Original Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(before, bins=20, color="#f97316", alpha=0.8, edgecolor="black")
        ax1.set_title(f"Original: {feature}")
        ax1.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig1)

    with col_chart2:
        st.subheader("Standard Scaled Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(after, bins=20, color="#06b6d4", alpha=0.8, edgecolor="black")
        ax2.set_title(f"Scaled: {feature}")
        ax2.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig2)

# =====================================================
# SGD REGRESSOR
# =====================================================
elif page == "SGD Regressor":
    st.title("🤖 Scikit-learn SGD Regressor Workspace")

    alpha = st.slider("Regularization Alpha", 0.000001, 1.0, 0.01, format="%.6f")

    model = SGDRegressor(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    # Metric Grid
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("MSE", f"{mse:.3f}")
    c3.metric("R² Score", f"{r2:.3f}")

    st.markdown("---")

    # Grid Layout for Charts: Coefficients vs Performance
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        }).sort_values("Coefficient")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color="#8b5cf6")
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig)

    with col_right:
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, pred, alpha=0.7, color="#ec4899")
        mn = min(y_test.min(), pred.min())
        mx = max(y_test.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=2)
        ax.set_xlabel("Actual BP")
        ax.set_ylabel("Predicted BP")
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig)

# =====================================================
# GRADIENT DESCENT
# =====================================================
elif page == "Gradient Descent":
    st.title("📈 Custom Gradient Descent Engine")

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

    st.markdown("---")
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slope (Weight)", f"{float(slope[0]):.3f}")
    c2.metric("Intercept", f"{intercept:.3f}")
    c3.metric("MSE", f"{mse:.3f}")
    c4.metric("R² Score", f"{r2:.3f}")

    st.markdown("---")

    # Grid Layout for Visualizations
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Cost vs Iteration")
        history_df = pd.DataFrame(history, columns=["Slope", "Intercept", "Cost"])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(history_df["Cost"], color="#ef4444", linewidth=2.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost (MSE)")
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    with col_g2:
        st.subheader("Fitted Regression Line (BMI)")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(X_train_new[:, 0], y_train, alpha=0.5, color="#3b82f6", label="Data")
        
        line_x = np.linspace(X_train_new[:, 0].min(), X_train_new[:, 0].max(), 100).reshape(-1, 1)
        line_y = (line_x @ slope + intercept).ravel()
        ax.plot(line_x[:, 0], line_y, color="#10b981", linewidth=3, label="Fit Line")
        
        ax.set_xlabel("BMI (Scaled)")
        ax.set_ylabel("Blood Pressure")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

# =====================================================
# 3D COST SURFACE
# =====================================================
elif page == "3D Cost Surface":
    st.title("🌋 3D Cost Surface Landscape")
    st.markdown("Visualizing the optimization path over the error surface for slope and intercept parameters.")

    bmi_idx = X.columns.get_loc("bmi")
    X_train_new = X_train_scaled[:, [bmi_idx]]

    gd = GradientDescent(alfa=0.1, iterations=100)
    gd.fit(X_train_new, y_train.values, track_history=True)

    slope = gd.m
    intercept = gd.c

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

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(M, C, Z, cmap="plasma", alpha=0.85, edgecolor="none")
    ax.scatter(
        float(slope[0]), intercept, 
        cost(float(slope[0]), intercept, X_train_new, y_train.values),
        color="cyan", s=100, label="Final Parameters"
    )

    ax.set_xlabel("Slope")
    ax.set_ylabel("Intercept")
    ax.set_zlabel("Cost")
    ax.legend()
    fig.colorbar(surface, shrink=0.5, aspect=10)

    st.pyplot(fig)

# =====================================================
# CONCLUSION
# =====================================================
else:
    st.title("✅ Summary & Architecture")

    st.markdown("""
    ### Key Takeaways
    - **Modular Design**: Leveraged an isolated external `GradientDescent.py` script containing a custom loop structure[cite: 1].
    - **Interactive Optimization**: Tracked convergence metrics step-by-step using internal state histories.
    - **Comparative Insight**: Evaluated custom scratch code against Scikit-Learn's native optimization algorithms seamlessly.
    """)