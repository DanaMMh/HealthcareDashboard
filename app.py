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

# default fallback filename the app used previously
DEFAULT_FILENAME = "patients.csv"
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


def find_csv_files() -> list[Path]:
    """Search for CSV files in sensible locations: repo root and data/ directories."""
    base = Path(__file__).resolve().parent
    candidates: list[Path] = []
    root_file = base / DEFAULT_FILENAME
    if root_file.exists():
        candidates.append(root_file)

    data_dir = base / "data"
    if data_dir.exists() and data_dir.is_dir():
        for p in sorted(data_dir.rglob("*.csv")):
            candidates.append(p)

    # also check data/liver specifically (download script target)
    liver_dir = base / "data" / "liver"
    if liver_dir.exists() and liver_dir.is_dir():
        for p in sorted(liver_dir.rglob("*.csv")):
            candidates.append(p)

    # dedupe while preserving order
    seen = set()
    unique = []
    for p in candidates:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)
    return unique


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

    csv_files = find_csv_files()

    st.title("Healthcare Analytics Dashboard")

    selected_df = None
    selected_path = None

    if csv_files:
        options = [str(p.relative_to(Path(__file__).resolve().parent)) for p in csv_files]
        options.append("Upload a local file")
        choice = st.selectbox("Select a dataset to load", options, index=0)
        if choice == "Upload a local file":
            uploaded = st.file_uploader("Upload a CSV file", type="csv")
            if uploaded is not None:
                try:
                    selected_df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"Could not read uploaded file: {e}")
                    st.stop()
        else:
            selected_path = Path(__file__).resolve().parent / choice
    else:
        st.info(
            "No dataset found in the repository. You can run scripts/download_kaggle.py to download the Kaggle dataset into data/, or upload a CSV file below."
        )
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded is not None:
            try:
                selected_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                st.stop()

    # If a path was selected, try to load it
    if selected_df is None and selected_path is not None:
        if not selected_path.exists():
            st.error(f"Dataset not found: {selected_path}")
            st.stop()
        try:
            selected_df = load_data(str(selected_path))
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    if selected_df is None:
        st.stop()

    df = selected_df

    if df.empty:
        st.warning("The dataset is empty.")
        st.stop()

    try:
        accuracy, test_records = evaluate_model(df)
    except (KeyError, ValueError) as error:
        st.error(f"Could not calculate model accuracy: {error}")
        # continue to show dataset preview even if model evaluation failed
        accuracy = None
        test_records = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        if accuracy is not None:
            st.metric("Test Accuracy", f"{accuracy:.1%}", help=f"Measured on {test_records} held-out records.")
        else:
            st.metric("Test Accuracy", "N/A", help="Model evaluation failed for this dataset.")

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
