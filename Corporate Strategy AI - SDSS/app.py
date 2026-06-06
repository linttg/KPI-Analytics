# =====================================================
# IMPORT LIBRARY
# =====================================================

from chatbot import ask_ai

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# SAFE IMPORT LSTM
# =====================================================

LSTM_AVAILABLE = True

try:

    from model_trainer import (
        train_lstm_model,
        plot_training,
        plot_prediction,
        plot_heatmap,
        calculate_vif,
        compare_models
    )

except Exception as e:

    LSTM_AVAILABLE = False
    lstm_error = str(e)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI KPI Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color:#f8fafc;
}

h1,h2,h3{
    color:#0f172a;
}

section[data-testid="stSidebar"]{
    background-color:#e2e8f0;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:20px;
    border:1px solid #cbd5e1;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Dashboard Configuration")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("## 🤖 Fine Tuning AI")

epochs = st.sidebar.slider(
    "Epoch",
    10,
    200,
    50
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [8, 16, 32, 64],
    index=1
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.1, 0.01, 0.001, 0.0001],
    index=2
)

# =====================================================
# LOAD DATA
# =====================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        st.sidebar.success(
            "✅ Dataset uploaded successfully"
        )

    except Exception as e:

        st.error(f"Error reading file: {e}")

        st.stop()

else:

    df = pd.DataFrame({

        "Bulan": [
            "Jan", "Feb", "Mar",
            "Apr", "Mei", "Jun"
        ],

        "Penjualan": [
            120, 135, 150,
            145, 170, 180
        ],

        "Komplain": [
            40, 35, 30,
            28, 20, 18
        ],

        "Profit": [
            15, 18, 20,
            17, 25, 28
        ]

    })

    st.sidebar.info(
        "📌 Using default sample dataset"
    )

# =====================================================
# HEADER
# =====================================================

st.title("📊 AI KPI Analytics Dashboard")

st.markdown("""
Interactive KPI Dashboard with:
- AI Forecasting
- Deep Learning Analytics
- Business Insight
- Research Recommendation
- AI Chatbot Assistant
""")

st.divider()

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =====================================================
# AUTO DATE PROCESSING
# =====================================================

if "Date" in df.columns:

    try:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df["Bulan"] = (
            df["Date"]
            .dt.strftime("%b")
        )

    except:
        pass

# =====================================================
# AUTO SALES COLUMN
# =====================================================

if "Penjualan" not in df.columns:

    if "Total Amount" in df.columns:

        sales_monthly = (
            df.groupby("Bulan")["Total Amount"]
            .sum()
            .reset_index()
        )

        sales_monthly.columns = [
            "Bulan",
            "Penjualan"
        ]

    else:

        numeric_cols = (
            df.select_dtypes(include=np.number)
            .columns
        )

        numeric_col = numeric_cols[0]

        sales_monthly = (
            df.groupby("Bulan")[numeric_col]
            .sum()
            .reset_index()
        )

        sales_monthly.columns = [
            "Bulan",
            "Penjualan"
        ]

else:

    if (
        "Bulan" in df.columns
        and df["Bulan"].duplicated().any()
    ):

        sales_monthly = (
            df.groupby("Bulan")["Penjualan"]
            .sum()
            .reset_index()
        )

    else:

        sales_monthly = df[
            ["Bulan", "Penjualan"]
        ].copy()

# =====================================================
# AUTO COMPLAINT COLUMN
# =====================================================

complaint_keywords = [
    "Komplain",
    "Complaint",
    "Complaints",
    "Customer Satisfaction"
]

found_complaint = None

for col in df.columns:

    if col in complaint_keywords:

        found_complaint = col
        break

# =====================================================
# CREATE COMPLAINT DATA
# =====================================================

if found_complaint is not None:

    if (
        "Bulan" in df.columns
        and df["Bulan"].duplicated().any()
    ):

        complaint_monthly = (
            df.groupby("Bulan")[found_complaint]
            .sum()
            .reset_index()
        )

        sales_monthly["Komplain"] = (
            complaint_monthly[
                found_complaint
            ].values
        )

    else:

        sales_monthly["Komplain"] = (
            df[found_complaint].values
        )

else:

    sales_values = (
        sales_monthly["Penjualan"]
        .values
    )

    normalized = (
        sales_values /
        sales_values.max()
    )

    generated_complaint = (
        50 - (normalized * 30)
    ).astype(int)

    sales_monthly["Komplain"] = (
        generated_complaint
    )

# =====================================================
# KPI DATA
# =====================================================

kpi_df = sales_monthly.copy()

# =====================================================
# KPI SUMMARY
# =====================================================

st.subheader("📊 KPI Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💰 Total Sales",
    f"{kpi_df['Penjualan'].sum():,.0f}"
)

c2.metric(
    "📈 Average Sales",
    f"{kpi_df['Penjualan'].mean():,.2f}"
)

c3.metric(
    "⚠️ Total Complaints",
    f"{kpi_df['Komplain'].sum():,.0f}"
)

growth = (
    (
        kpi_df["Penjualan"].iloc[-1]
        -
        kpi_df["Penjualan"].iloc[0]
    )
    /
    kpi_df["Penjualan"].iloc[0]
) * 100

c4.metric(
    "🚀 Growth Rate",
    f"{growth:.2f}%"
)

st.divider()

# =====================================================
# VISUALIZATION
# =====================================================

st.subheader("📈 KPI Visualization")

col1, col2 = st.columns(2)

with col1:

    fig1, ax1 = plt.subplots(
        figsize=(7, 4)
    )

    ax1.plot(
        kpi_df["Bulan"],
        kpi_df["Penjualan"],
        marker="o",
        linewidth=3
    )

    ax1.set_title(
        "Sales Trend"
    )

    ax1.grid(alpha=0.3)

    st.pyplot(fig1)

with col2:

    fig2, ax2 = plt.subplots(
        figsize=(7, 4)
    )

    ax2.bar(
        kpi_df["Bulan"],
        kpi_df["Komplain"]
    )

    ax2.set_title(
        "Customer Complaints"
    )

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

# =====================================================
# EDA
# =====================================================

st.subheader("📊 Exploratory Data Analysis")

eda1, eda2 = st.columns(2)

with eda1:

    st.write("### Statistical Summary")
    st.dataframe(df.describe())

with eda2:

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

st.write("### Correlation Matrix")

st.write(
    "### Correlation Heatmap"
)

fig_heatmap = plot_heatmap(df)

st.pyplot(fig_heatmap)

corr = df.corr(numeric_only=True)

st.dataframe(corr)

st.write(
    "### Multicollinearity (VIF)"
)

vif_df = calculate_vif(df)

st.dataframe(
    vif_df,
    use_container_width=True
)

# =====================================================
# AI SALES PREDICTION
# =====================================================

st.subheader("🤖 AI Sales Prediction")

avg_growth = (
    kpi_df["Penjualan"]
    .pct_change()
    .mean()
)

next_sales = (
    kpi_df["Penjualan"].iloc[-1]
    * (1 + avg_growth)
)

st.success(f"""
Predicted Next Month Sales:
{next_sales:,.2f}
""")

# =====================================================
# AI BUSINESS INSIGHT
# =====================================================

st.subheader("🧠 AI Business Insight")

complaint_change = (
    kpi_df["Komplain"].iloc[-1]
    -
    kpi_df["Komplain"].iloc[0]
)

if complaint_change > 0:

    st.warning(f"""
Customer complaints increased by
{complaint_change:,.0f}.

Recommendation:
Improve customer experience and service quality.
""")

else:

    st.success(f"""
Sales growth reached {growth:.2f}%.

Business performance shows positive growth trends.
""")

# =====================================================
# RESEARCH RECOMMENDATION
# =====================================================

st.subheader("🔬 Research Recommendation")

def get_research_papers(keyword):

    url = (
        f"https://api.openalex.org/works"
        f"?search={keyword}"
        f"&per-page=3"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            return data.get(
                "results",
                []
            )

        return []

    except:

        return []

papers = get_research_papers(
    "business analytics"
)

if len(papers) > 0:

    for paper in papers:

        title = paper.get(
            "title",
            "No Title"
        )

        year = paper.get(
            "publication_year",
            "Unknown"
        )

        cited = paper.get(
            "cited_by_count",
            0
        )

        st.markdown(f"""
### 📘 {title}

📅 Year: {year}  
📖 Citations: {cited}
""")

        st.markdown("---")

# =====================================================
# LSTM MODEL
# =====================================================

st.subheader(
    "🏆 Machine Learning Comparison"
)

comparison_df = compare_models(df)

st.dataframe(
    comparison_df,
    use_container_width=True
)

st.subheader("🤖 Deep Learning Forecasting")

if not LSTM_AVAILABLE:

    st.warning(f"""
TensorFlow / LSTM tidak tersedia di Python 3.13.

Error:
{lstm_error}

Gunakan Python 3.11 untuk menjalankan model Deep Learning.
""")

else:

    try:

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if len(numeric_df.columns) >= 2:

            with st.spinner(
                "Training LSTM Model..."
            ):

                model, history, results, y_test, y_pred = (
                    train_lstm_model(
                        numeric_df,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )
                )

            st.subheader("📊 Model Evaluation")

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "RMSE",
                results["RMSE"]
            )

            e2.metric(
                "MSE",
                results["MSE"]
            )

            e3.metric(
                "MAPE",
                results["MAPE"]
            )

            e4.metric(
                "R2 Score",
                results["R2 Score"]
            )

            st.subheader("📉 Training Loss")

            fig_loss = plot_training(
                history
            )

            st.pyplot(fig_loss)

            st.subheader(
                "📈 Actual vs Prediction"
            )

            fig_pred = plot_prediction(
                y_test,
                y_pred
            )

            st.pyplot(fig_pred)

        else:

            st.warning("""
Dataset membutuhkan minimal
2 kolom numerik untuk training model.
""")

    except Exception as e:

        st.error(
            f"Training model error: {e}"
        )

# =====================================================
# AI CHATBOT
# =====================================================

st.subheader("🤖 AI Chatbot Assistant")

user_question = st.text_area(
    "Ask AI about your business data"
)

if st.button("Generate AI Insight"):

    if user_question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "AI is analyzing..."
        ):

            try:

                prompt = f"""
                KPI Dataset Summary:

                Total Sales:
                {kpi_df['Penjualan'].sum()}

                Average Sales:
                {kpi_df['Penjualan'].mean()}

                Total Complaints:
                {kpi_df['Komplain'].sum()}

                User Question:
                {user_question}
                """

                answer = ask_ai(prompt)

                st.success(answer)

            except Exception as e:

                st.error(
                    f"Chatbot Error: {e}"
                )