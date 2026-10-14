from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "patients.csv"
TARGET_COLUMN = "Result"


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner="Training a baseline model...")
def evaluate_model(df: pd.DataFrame) -> tuple[float, int]:
    labeled = df.dropna(subset=[TARGET_COLUMN]).copy()
    features = labeled.drop(columns=[TARGET_COLUMN])
    target = labeled[TARGET_COLUMN]

    if target.nunique() < 2 or target.value_counts().min() < 2:
        raise ValueError("At least two classes with two or more records each are required.")

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    transformers = []

    if numeric_columns:
        numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    model = make_pipeline(
        ColumnTransformer(transformers=transformers),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(x_train, y_train)
    return accuracy_score(y_test, model.predict(x_test)), len(y_test)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns: list[str] = []
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def main() -> None:
    st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #f4f8fc; }
        [data-testid="stHeader"] { background-color: #f4f8fc; }
        h1, h2, h3 { color: #123b66; }
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #d7e6f5;
            border-left: 4px solid #2878c8;
            border-radius: 8px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #45627f; }
        [data-testid="stMetricValue"] { color: #123b66; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    df = load_data(str(DATA_PATH))
    if df.empty:
        st.warning("The dataset is empty.")
        st.stop()

    st.title("Healthcare Analytics Dashboard")

    try:
        accuracy, test_records = evaluate_model(df)
    except (KeyError, ValueError) as error:
        st.error(f"Could not calculate model accuracy: {error}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Test Accuracy", f"{accuracy:.1%}", help=f"Measured on {test_records} held-out records.")
    st.caption(
        "Accuracy is from a baseline logistic-regression model evaluated on a stratified "
        "20% test split. It is an estimate, not evidence of clinical validity."
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all").T)

    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        selected_col = st.selectbox("Select a metric to visualize", numeric_cols, index=0)
        chart_data = pd.to_numeric(df[selected_col], errors="coerce").dropna()

        fig, ax = plt.subplots()
        chart_data.hist(bins=min(20, max(5, len(chart_data) // 10)), ax=ax, color="#2878c8")
        ax.set_title(f"Distribution of {selected_col}", color="#123b66")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns are available for visualization.")


if __name__ == "__main__":
    main()
