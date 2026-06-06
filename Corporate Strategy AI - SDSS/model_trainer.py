# =====================================================
# IMPORT LIBRARY
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.model_selection import (
    cross_val_score
)

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)

from xgboost import XGBRegressor

import seaborn as sns

# =====================================================
# LOAD DATA
# =====================================================

def load_data(file):

    try:

        if file.name.endswith(".csv"):

            df = pd.read_csv(file)

        else:

            df = pd.read_excel(file)

        return df

    except Exception as e:

        print(e)

        return pd.DataFrame()

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_data(df):

    df = df.drop_duplicates()

    df = df.ffill()

    numeric_df = df.select_dtypes(
        include=np.number
    )

    return numeric_df

# =====================================================
# KPI
# =====================================================

# =====================================================
# VIF MULTICOLLINEARITY
# =====================================================

def calculate_vif(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df.columns) < 2:

        return pd.DataFrame()

    vif = pd.DataFrame()

    vif["Variable"] = numeric_df.columns

    vif["VIF"] = [

        variance_inflation_factor(
            numeric_df.values,
            i
        )

        for i in range(
            numeric_df.shape[1]
        )
    ]

    return vif

# =====================================================
# DNN TRAINING
# =====================================================

def train_lstm_model(

    df,

    epochs=100,

    batch_size=16,

    learning_rate=0.001
):

    # =================================================
    # NUMERIC DATA
    # =================================================

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df.columns) < 2:

        raise ValueError(
            "Dataset minimal harus punya 2 kolom numerik."
        )

    # =================================================
    # TARGET
    # =================================================

    target_col = numeric_df.columns[-1]

    X = numeric_df.drop(
        columns=[target_col]
    )

    y = numeric_df[target_col]

    # =================================================
    # SCALING
    # =================================================

    scaler_x = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)

    y_scaled = scaler_y.fit_transform(
        y.values.reshape(-1,1)
    ).ravel()

    # =================================================
    # SPLIT
    # =================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled,
        y_scaled,

        test_size=0.2,

        random_state=42
    )

    # =================================================
    # DNN MODEL
    # =================================================

    model = MLPRegressor(

        hidden_layer_sizes=(128,64,32),

        learning_rate_init=learning_rate,

        batch_size=batch_size,

        max_iter=epochs,

        activation="relu",

        solver="adam",

        random_state=42
    )

    # =================================================
    # TRAIN
    # =================================================

    model.fit(
        X_train,
        y_train
    )

    # =================================================
    # PREDICTION
    # =================================================

    y_pred = model.predict(X_test)

    # =================================================
    # INVERSE SCALE
    # =================================================

    y_test_actual = scaler_y.inverse_transform(
        y_test.reshape(-1,1)
    )

    y_pred_actual = scaler_y.inverse_transform(
        y_pred.reshape(-1,1)
    )

    # =================================================
    # METRICS
    # =================================================

    rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            y_pred_actual
        )
    )

    mse = mean_squared_error(
        y_test_actual,
        y_pred_actual
    )

    mape = mean_absolute_percentage_error(
        y_test_actual,
        y_pred_actual
    )

    r2 = r2_score(
        y_test_actual,
        y_pred_actual
    )

    results = {

        "RMSE":
        round(rmse,4),

        "MSE":
        round(mse,4),

        "MAPE":
        round(mape,4),

        "R2 Score":
        round(r2,4)
    }

    # =================================================
    # HISTORY
    # =================================================

    history = {

        "loss":
        model.loss_curve_
    }

    return (

        model,
        history,
        results,
        y_test_actual,
        y_pred_actual
    )

# =====================================================
# PLOT TRAINING
# =====================================================

def plot_training(history):

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    # =================================================
    # TRAIN LOSS
    # =================================================

    ax.plot(

        history["loss"],

        label="Training Loss",

        linewidth=3
    )

    # =================================================
    # STYLE
    # =================================================

    ax.set_title(
        "Training Performance",
        fontsize=18
    )

    ax.set_xlabel(
        "Iteration",
        fontsize=12
    )

    ax.set_ylabel(
        "Loss",
        fontsize=12
    )

    ax.grid(alpha=0.3)

    ax.legend()

    return fig

# =====================================================
# ACTUAL VS PREDICTION
# =====================================================

def plot_prediction(
    y_test,
    y_pred
):

    y_test = np.array(y_test).flatten()

    y_pred = np.array(y_pred).flatten()

    sorted_idx = np.argsort(y_test)

    y_test_sorted = y_test[sorted_idx]

    y_pred_sorted = y_pred[sorted_idx]

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.plot(
        y_test_sorted,
        label="Actual",
        linewidth=3
    )

    ax.plot(
        y_pred_sorted,
        label="Prediction",
        linewidth=3
    )

    ax.set_title(
        "Actual vs Prediction"
    )

    ax.set_xlabel(
        "Sorted Data Index"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    return fig


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def plot_heatmap(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.heatmap(

        corr,

        annot=True,

        cmap="coolwarm",

        fmt=".2f",

        linewidths=0.5,

        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=18
    )

    return fig

# =====================================================
# MACHINE LEARNING COMPARISON
# =====================================================

def compare_models(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    target_col = numeric_df.columns[-1]

    X = numeric_df.drop(
        columns=[target_col]
    )

    y = numeric_df[target_col]

    models = {

        "Linear Regression":
        LinearRegression(),

        "Ridge":
        Ridge(alpha=1.0),

        "Lasso":
        Lasso(alpha=0.01),

        "XGBoost":
        XGBRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():

        scores = cross_val_score(

            model,

            X,

            y,

            cv=5,

            scoring="r2"
        )

        results.append({

            "Model":
            name,

            "CV R2":
            round(
                scores.mean(),
                4
            )
        })

    return pd.DataFrame(
        results
    )
