
import os
import time
from datetime import datetime
from html import escape
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from evaluator import evaluate_answer
from database import (
    save_evaluation_run,
    get_evaluation_history,
    get_evaluation_run,
)


# ==============================================
# CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="SOLVYN | AI Evaluation Workspace",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

GOLD = "#E6BC58"
BG = "#0D0E11"
PANEL = "#191A1F"
TEXT = "#F3F3F5"
MUTED = "#A6A7B2"

MODEL_NAME = "openai/gpt-oss-20b"

SAMPLE_DATA = [
    {
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    {
        "question": "What does LLM stand for?",
        "expected": "Large Language Model",
    },
    {
        "question": "What is the capital of Pakistan?",
        "expected": "Islamabad",
    },
    {
        "question": "What does RAG stand for in AI?",
        "expected": "Retrieval Augmented Generation",
    },
    {
        "question": "What programming language is Streamlit built with?",
        "expected": "Python",
    },
]

DEFAULT_STATE = {
    "page": "Overview",
    "dataset": [dict(row) for row in SAMPLE_DATA],
    "results": [],
    "run_model": "",
    "run_date": "",
    "last_run_id": None,
    "editor_version": 0,
    "notice": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==============================================
# STYLING
# ==============================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0D0E11;
        color: #F3F3F5;
    }

    .block-container {
        max-width: 1600px;
        padding: 2.2rem 2.4rem 5rem;
    }

    [data-testid="stHeader"] {
        background: #0D0E11;
    }

    [data-testid="stSidebar"] {
        background: #141518;
        border-right: 1px solid #303137;
    }

    h1, h2, h3 {
        color: #F3F3F5;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 2.15rem !important;
        font-weight: 700 !important;
    }

    p, label {
        line-height: 1.55;
    }

    hr {
        border-color: #303137 !important;
    }

    /* GOLD ACTION BUTTONS */

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background: #E6BC58 !important;
        border: 1px solid #E6BC58 !important;
        color: #111111 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }

    div.stButton > button[kind="primary"] *,
    div.stDownloadButton > button[kind="primary"] * {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        opacity: 1 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background: #F5D483 !important;
        border-color: #F5D483 !important;
    }

    div.stButton > button[kind="secondary"] {
        background: #1D1E23;
        border: 1px solid #36373E;
        color: #F3F3F5;
        border-radius: 8px;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: #292A30;
        border-color: #8F7747;
        color: #FFFFFF;
    }

    /* SIDEBAR NAVIGATION */

    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        min-height: 42px;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid transparent;
        color: #B7B7C0;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #40371F !important;
        border: 1px solid #61502D !important;
        color: #F5D483 !important;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] * {
        color: #F5D483 !important;
        -webkit-text-fill-color: #F5D483 !important;
    }

    /* INPUTS AND TABLES */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #1D1E23;
        border-color: #383940;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        border: 1px solid #34353B;
        border-radius: 9px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: #191A1F;
        border: 1px solid #34353B;
        border-radius: 9px;
        padding: 15px;
    }

    [data-testid="stMetricValue"] {
        color: #F3F3F5;
    }

    /* BRAND */

    .solvyn-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 6px;
    }

    .solvyn-symbol {
        color: #E6BC58;
        font-size: 28px;
    }

    .solvyn-name {
        color: #FFFFFF;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .solvyn-subtitle {
        color: #92929C;
        font-size: 10px;
        letter-spacing: 1.4px;
        font-weight: 700;
        margin-bottom: 23px;
    }

    .breadcrumb {
        color: #9696A0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 13px;
    }

    .page-desc {
        color: #A6A7B2;
        font-size: 14px;
        margin: -6px 0 25px;
    }

    .gold-line {
        width: 48px;
        height: 2px;
        background: #E6BC58;
        margin-bottom: 25px;
    }

    .section-heading {
        color: #F3F3F5;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 15px;
    }

    /* METRIC CARDS */

    .metric-card {
        background: #191A1F;
        border: 1px solid #34353B;
        border-radius: 9px;
        padding: 21px;
        min-height: 112px;
    }

    .metric-label {
        font-size: 11px;
        color: #AAAAB5;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #F5F5F6;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -1px;
    }

    /* DETAIL PANELS */

    .detail-panel {
        background: #191A1F;
        border: 1px solid #34353B;
        border-radius: 9px;
        padding: 19px;
        margin: 10px 0 20px;
    }

    .detail-title {
        color: #F5F5F6;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .detail-body {
        color: #B5B5BF;
        font-size: 13px;
        line-height: 1.7;
    }

    .answer-box {
        background: #202126;
        border: 1px solid #37383F;
        color: #F0F0F3;
        border-radius: 8px;
        padding: 17px;
        font-size: 14px;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        margin: 9px 0 18px;
    }

    .empty-state {
        background: #191A1F;
        border: 1px solid #34353B;
        border-radius: 10px;
        padding: 45px 22px;
        text-align: center;
        margin: 20px 0;
    }

    .empty-icon {
        color: #E6BC58;
        font-size: 30px;
        margin-bottom: 12px;
    }

    .empty-title {
        color: #F3F3F5;
        font-size: 19px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .empty-description {
        color: #A6A7B2;
        font-size: 13px;
    }

    .api-status {
        background: #193225;
        color: #93DDB0;
        border: 1px solid #30543F;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 12px;
        margin-top: 10px;
    }

    .api-missing {
        background: #362B1A;
        color: #F3CE83;
        border: 1px solid #65502B;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 12px;
        margin-top: 10px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .metric-card {
            padding: 14px;
        }

        .metric-value {
            font-size: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================
# HELPERS
# ==============================================

def get_api_key():
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        return None


def navigate(page):
    st.session_state.page = page
    st.rerun()


def page_header(section, title, description):
    st.markdown(
        f'<div class="breadcrumb">WORKSPACE / '
        f'{escape(section.upper())}</div>',
        unsafe_allow_html=True,
    )

    st.title(title)

    st.markdown(
        f'<div class="page-desc">'
        f'{escape(description)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="gold-line"></div>',
        unsafe_allow_html=True,
    )


def heading(title):
    st.markdown(
        f'<div class="section-heading">'
        f'{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {escape(str(label))}
            </div>
            <div class="metric-value">
                {escape(str(value))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title, description):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">◇</div>
            <div class="empty-title">
                {escape(title)}
            </div>
            <div class="empty-description">
                {escape(description)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_box(value):
    st.markdown(
        f'<div class="answer-box">'
        f'{escape(str(value))}</div>',
        unsafe_allow_html=True,
    )


def safe_history():
    try:
        return get_evaluation_history()
    except Exception:
        st.error("Could not load evaluation history.")
        return []


def get_latest_run():
    # Session data takes priority.
    if st.session_state.results:
        return {
            "id": st.session_state.last_run_id,
            "model": st.session_state.run_model,
            "created_at": st.session_state.run_date,
            "results": st.session_state.results,
        }

    history = safe_history()

    if not history:
        return None

    try:
        return get_evaluation_run(history[0]["id"])
    except Exception:
        return None


def show_metrics(results):
    if not results:
        return

    df = pd.DataFrame(results)

    values = [
        ("EXACT MATCH", f"{df['exact_match'].mean() * 100:.1f}%"),
        ("TOKEN F1", f"{df['token_f1'].mean() * 100:.1f}%"),
        (
            "SEMANTIC SIMILARITY",
            f"{df['semantic_similarity'].mean() * 100:.1f}%",
        ),
        ("AVG. LATENCY", f"{df['latency'].mean():.2f}s"),
    ]

    columns = st.columns(4)

    for column, (label, value) in zip(columns, values):
        with column:
            metric_card(label, value)


def show_empty_metrics():
    columns = st.columns(4)

    for column, label in zip(
        columns,
        [
            "EXACT MATCH",
            "TOKEN F1",
            "SEMANTIC SIMILARITY",
            "AVG. LATENCY",
        ],
    ):
        with column:
            metric_card(label, "—")


def results_table(results):
    df = pd.DataFrame(results)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def download_results(results, filename, key):
    df = pd.DataFrame(results)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        type="primary",
        key=key,
    )


# ==============================================
# ANALYTICS HELPERS
# ==============================================

def chart_layout(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="DM Sans"),
        margin=dict(l=20, r=20, t=35, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.15,
        ),
        xaxis=dict(
            gridcolor="#2B2C32",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#2B2C32",
            zeroline=False,
        ),
    )

    return fig


def metric_chart(results):
    df = pd.DataFrame(results)

    metrics = [
        ("Exact Match", df["exact_match"].mean() * 100),
        ("Token F1", df["token_f1"].mean() * 100),
        (
            "Semantic Similarity",
            df["semantic_similarity"].mean() * 100,
        ),
        (
            "Keyword Coverage",
            df["keyword_coverage"].mean() * 100,
        ),
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[name for name, _ in metrics],
            y=[value for _, value in metrics],
            marker_color=GOLD,
            text=[f"{value:.1f}%" for _, value in metrics],
            textposition="outside",
        )
    )

    fig.update_yaxes(range=[0, 110], title="Score (%)")

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True,
    )


def latency_chart(results):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(results) + 1)),
            y=[row["latency"] for row in results],
            mode="lines+markers",
            line=dict(color=GOLD, width=2),
            marker=dict(size=8),
            name="Latency",
        )
    )

    fig.update_xaxes(title="Test case")
    fig.update_yaxes(title="Seconds")

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True,
    )


def show_analytics(results):
    if not results:
        empty_state(
            "No analytics available",
            "Run an evaluation to generate performance charts.",
        )
        return

    show_metrics(results)

    st.divider()

    left, right = st.columns(2)

    with left:
        heading("Evaluation metrics")
        metric_chart(results)

    with right:
        heading("Response latency")
        latency_chart(results)

    df = pd.DataFrame(results)

    st.divider()
    heading("Execution summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Test cases", len(df))
    c2.metric("Total tokens", int(df["tokens"].sum()))
    c3.metric(
        "Total model latency",
        f"{df['latency'].sum():.2f}s",
    )

    st.caption(
        "Scores are similarity measurements, not independently "
        "verified factual accuracy."
    )


# ==============================================
# DATASET VALIDATION
# ==============================================

def validate_dataset(dataframe):
    required = {"question", "expected"}

    if not required.issubset(dataframe.columns):
        raise ValueError(
            "CSV must contain 'question' and 'expected' columns."
        )

    clean = dataframe[["question", "expected"]].copy()

    clean = clean.dropna(
        subset=["question", "expected"]
    )

    for column in ["question", "expected"]:
        clean[column] = (
            clean[column].astype(str).str.strip()
        )

    clean = clean[
        (clean["question"] != "")
        & (clean["expected"] != "")
    ]

    if clean.empty:
        raise ValueError(
            "Dataset must contain at least one valid test case."
        )

    if len(clean) > 100:
        raise ValueError(
            "Maximum 100 test cases are allowed per dataset."
        )

    return clean.to_dict(orient="records")


def clear_current_results():
    st.session_state.results = []
    st.session_state.run_model = ""
    st.session_state.run_date = ""
    st.session_state.last_run_id = None


# ==============================================
# GROQ INFERENCE
# ==============================================

def generate_answer(client, question, model_name):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer accurately and concisely. "
                    "For short factual questions, "
                    "provide direct answers."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0,
    )

    latency = time.perf_counter() - start

    answer = response.choices[0].message.content or ""

    usage = response.usage

    tokens = usage.total_tokens if usage else 0

    return answer, round(latency, 3), tokens


def run_evaluation(dataset, model_name):
    key = get_api_key()

    if not key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=key)

    results = []

    progress = st.progress(
        0,
        text="Starting evaluation...",
    )

    try:
        for index, case in enumerate(dataset):
            try:
                actual, latency, tokens = generate_answer(
                    client,
                    case["question"],
                    model_name,
                )

                scores = evaluate_answer(
                    case["expected"],
                    actual,
                )

            except Exception as error:
                raise RuntimeError(
                    f"Test case {index + 1} failed. "
                    "Check your API connection, limits, "
                    "and input data."
                ) from error

            results.append(
                {
                    "question": case["question"],
                    "expected": case["expected"],
                    "actual": actual,
                    "exact_match": scores["exact_match"],
                    "token_f1": scores["token_f1"],
                    "keyword_coverage": scores[
                        "keyword_coverage"
                    ],
                    "semantic_similarity": scores[
                        "semantic_similarity"
                    ],
                    "latency": latency,
                    "tokens": tokens,
                }
            )

            progress.progress(
                (index + 1) / len(dataset),
                text=(
                    f"Evaluated {index + 1} "
                    f"of {len(dataset)} test cases"
                ),
            )

    finally:
        progress.empty()

    return results


# ==============================================
# SIDEBAR
# ==============================================

with st.sidebar:
    st.markdown(
        """
        <div class="solvyn-brand">
            <span class="solvyn-symbol">◆</span>
            <span class="solvyn-name">SOLVYN</span>
        </div>
        <div class="solvyn-subtitle">
            AI EVALUATION WORKSPACE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    navigation = [
        ("Overview", "⌂  Overview"),
        ("Evaluations", "▷  Evaluations"),
        ("Datasets", "▤  Datasets"),
        ("Analytics", "▥  Analytics"),
        ("History", "◷  History"),
        ("Compare", "⇄  Compare Runs"),
        ("Reports", "▧  Reports"),
    ]

    for name, label in navigation:
        selected = st.session_state.page == name

        if st.button(
            label,
            key=f"nav_{name}",
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            navigate(name)

    st.divider()

    st.markdown(
        '<div class="breadcrumb">MODEL CONFIGURATION</div>',
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Model",
        [MODEL_NAME],
    )

    if get_api_key():
        st.markdown(
            '<div class="api-status">● API key configured</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="api-missing">● API key missing</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("SOLVYN / VERSION 1.9")


# ==============================================
# GLOBAL NOTICES
# ==============================================

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""


# ==============================================
# OVERVIEW PAGE
# ==============================================

if st.session_state.page == "Overview":
    page_header(
        "Overview",
        "Evaluation overview",
        "Monitor model quality, similarity and execution performance.",
    )

    latest = get_latest_run()

    if latest and latest["results"]:
        show_metrics(latest["results"])

        st.write("")
        heading("Latest evaluation run")

        st.caption(
            f"Model: {latest['model']} · "
            f"Run ID: {latest['id'] or 'Not saved'}"
        )

        results_table(latest["results"])

        if st.button("Inspect run", type="primary"):
            navigate("History")

    else:
        show_empty_metrics()

        empty_state(
            "No evaluations yet",
            "Run your first test dataset to see results.",
        )

        if st.button(
            "Open Evaluations",
            type="primary",
        ):
            navigate("Evaluations")

    st.divider()

    heading("Evaluation methodology")

    methods = [
        (
            "Exact Match",
            "Normalized answers must match exactly.",
        ),
        (
            "Token F1",
            "Measures overlapping word precision and recall.",
        ),
        (
            "Semantic Similarity",
            "Compares text embeddings using cosine similarity.",
        ),
        (
            "Keyword Coverage",
            "Measures expected unique-word coverage.",
        ),
    ]

    columns = st.columns(4)

    for column, (title, description) in zip(
        columns, methods
    ):
        with column:
            st.markdown(
                f"""
                <div class="detail-panel">
                    <div class="detail-title">
                        {escape(title)}
                    </div>
                    <div class="detail-body">
                        {escape(description)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "These metrics do not independently verify factual "
        "correctness or detect hallucinations."
    )


# ==============================================
# EVALUATIONS PAGE
# ==============================================

elif st.session_state.page == "Evaluations":
    page_header(
        "Evaluations",
        "Model evaluations",
        "Run your dataset and inspect individual model responses.",
    )

    left, right = st.columns([4, 1])

    with left:
        st.markdown(
            f"**Active dataset:** "
            f"{len(st.session_state.dataset)} test cases"
        )
        st.caption(f"Target model: {MODEL_NAME}")

    with right:
        clicked = st.button(
            "Run evaluation",
            type="primary",
            use_container_width=True,
        )

    if clicked:
        if not get_api_key():
            st.error("Add your GROQ_API_KEY to .env.")

        elif not st.session_state.dataset:
            st.error("Dataset is empty.")

        else:
            try:
                results = run_evaluation(
                    st.session_state.dataset,
                    MODEL_NAME,
                )

                # Save before announcing success.
                run_id = save_evaluation_run(
                    MODEL_NAME,
                    results,
                )

                st.session_state.results = results
                st.session_state.run_model = MODEL_NAME
                st.session_state.run_date = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )
                st.session_state.last_run_id = run_id

                st.success(
                    f"Evaluation saved as Run #{run_id}."
                )

            except RuntimeError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "Could not complete or save evaluation. "
                    "Check the terminal for more information."
                )

    st.divider()

    results = st.session_state.results

    if not results:
        empty_state(
            "Your evaluation workspace is ready",
            "Run a dataset to generate evaluation results.",
        )

    else:
        show_metrics(results)

        st.write("")
        heading("Test case explorer")

        list_col, answer_col, score_col = st.columns(
            [1.2, 1.8, 1],
            gap="medium",
        )

        with list_col:
            st.markdown("**Test cases**")

            selected = st.radio(
                "Select test case",
                range(len(results)),
                format_func=lambda i: (
                    f"#{i + 1:03d} · "
                    f"{results[i]['question'][:38]}"
                ),
                label_visibility="collapsed",
            )

        case = results[selected]

        with answer_col:
            st.markdown(
                f"**Test case #{selected + 1:03d}**"
            )

            st.caption("QUESTION")
            answer_box(case["question"])

            st.caption("MODEL RESPONSE")
            answer_box(case["actual"])

            st.caption("EXPECTED ANSWER")
            answer_box(case["expected"])

        with score_col:
            st.markdown("**Evaluation details**")

            for label, key in [
                ("Exact Match", "exact_match"),
                ("Token F1", "token_f1"),
                ("Keyword Coverage", "keyword_coverage"),
                ("Semantic Similarity", "semantic_similarity"),
            ]:
                st.metric(
                    label,
                    f"{case[key] * 100:.1f}%",
                )

            st.caption(
                f"Latency: {case['latency']:.2f}s"
            )
            st.caption(
                f"Tokens: {case['tokens']}"
            )


# ==============================================
# DATASETS PAGE
# ==============================================

elif st.session_state.page == "Datasets":
    page_header(
        "Datasets",
        "Test datasets",
        "Manage reference questions or import a custom CSV dataset.",
    )

    heading("Import CSV dataset")

    st.caption(
        "Required columns: question, expected. "
        "Maximum 100 valid rows."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded is not None:
        if st.button("Import dataset", type="primary"):
            try:
                # Simple upload size safeguard.
                if uploaded.size > 2 * 1024 * 1024:
                    raise ValueError(
                        "CSV must be smaller than 2 MB."
                    )

                content = uploaded.getvalue().decode(
                    "utf-8-sig"
                )

                dataframe = pd.read_csv(
                    StringIO(content)
                )

                imported = validate_dataset(dataframe)

                st.session_state.dataset = imported
                st.session_state.editor_version += 1
                clear_current_results()

                st.session_state.notice = (
                    f"Imported {len(imported)} test cases."
                )
                st.rerun()

            except (ValueError, UnicodeError, pd.errors.ParserError) as error:
                st.error(str(error))

    st.divider()

    heading("Active dataset")

    source = pd.DataFrame(
        st.session_state.dataset,
        columns=["question", "expected"],
    )

    edited = st.data_editor(
        source,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"editor_{st.session_state.editor_version}",
    )

    if st.button("Save dataset", type="primary"):
        try:
            clean = validate_dataset(edited)

            st.session_state.dataset = clean
            clear_current_results()

            st.success("Dataset saved successfully.")

        except ValueError as error:
            st.error(str(error))

    st.divider()

    st.download_button(
        "Download dataset template",
        data=pd.DataFrame(
            SAMPLE_DATA
        ).to_csv(index=False).encode("utf-8"),
        file_name="solvyn_dataset_template.csv",
        mime="text/csv",
    )

    st.caption(
        "Active datasets are session-based. "
        "Completed evaluation runs are stored in SQLite."
    )


# ==============================================
# ANALYTICS PAGE
# ==============================================

elif st.session_state.page == "Analytics":
    page_header(
        "Analytics",
        "Performance analytics",
        "Explore evaluation scores, latency and token usage.",
    )

    latest = get_latest_run()

    if latest:
        st.caption(
            f"Showing Run #{latest['id']} · "
            f"{latest['model']}"
        )

        show_analytics(latest["results"])

    else:
        empty_state(
            "No analytics yet",
            "Run an evaluation to unlock analytics.",
        )


# ==============================================
# HISTORY PAGE
# ==============================================

elif st.session_state.page == "History":
    page_header(
        "History",
        "Evaluation history",
        "Inspect and export previously saved evaluation runs.",
    )

    history = safe_history()

    if not history:
        empty_state(
            "No saved runs",
            "Complete an evaluation to create your history.",
        )

    else:
        heading("Saved evaluation runs")

        st.dataframe(
            pd.DataFrame(history),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        run_id = st.selectbox(
            "Select a run",
            [row["id"] for row in history],
            format_func=lambda value: f"Run #{value}",
        )

        run = get_evaluation_run(run_id)

        if run:
            heading(f"Run #{run_id}")

            st.caption(
                f"Model: {run['model']} · "
                f"Created: {run['created_at']}"
            )

            if run["results"]:
                show_metrics(run["results"])

                st.write("")
                results_table(run["results"])

                download_results(
                    run["results"],
                    f"solvyn_run_{run_id}.csv",
                    f"download_history_{run_id}",
                )


# ==============================================
# RUN COMPARISON PAGE
# ==============================================

elif st.session_state.page == "Compare":
    page_header(
        "Compare",
        "Run comparison",
        "Compare two saved evaluations side by side.",
    )

    history = safe_history()

    # Exclude the sample test-model database record.
    real_runs = [
        row for row in history
        if row["model"] != "test-model"
        and row["test_cases"] > 0
    ]

    if len(real_runs) < 2:
        empty_state(
            "Two real runs required",
            "Complete at least two Groq evaluations to compare results.",
        )

    else:
        choices = {
            row["id"]: row for row in real_runs
        }

        ids = list(choices.keys())

        left, right = st.columns(2)

        with left:
            id_a = st.selectbox(
                "Run A",
                ids,
                index=1,
                format_func=lambda value: (
                    f"Run #{value} · "
                    f"{choices[value]['model']}"
                ),
                key="compare_a",
            )

        with right:
            id_b = st.selectbox(
                "Run B",
                ids,
                index=0,
                format_func=lambda value: (
                    f"Run #{value} · "
                    f"{choices[value]['model']}"
                ),
                key="compare_b",
            )

        if id_a == id_b:
            st.warning(
                "Select two different evaluation runs."
            )

        else:
            run_a = get_evaluation_run(id_a)
            run_b = get_evaluation_run(id_b)

            if run_a and run_b:
                df_a = pd.DataFrame(run_a["results"])
                df_b = pd.DataFrame(run_b["results"])

                metrics = [
                    ("Exact Match", "exact_match"),
                    ("Token F1", "token_f1"),
                    ("Semantic Similarity", "semantic_similarity"),
                    ("Keyword Coverage", "keyword_coverage"),
                ]

                comparison = []

                for label, key in metrics:
                    a = df_a[key].mean() * 100
                    b = df_b[key].mean() * 100

                    comparison.append({
                        "Metric": label,
                        f"Run #{id_a}": round(a, 2),
                        f"Run #{id_b}": round(b, 2),
                        "Delta (B - A)": round(b - a, 2),
                    })

                heading("Metric comparison")

                comparison_df = pd.DataFrame(comparison)

                st.dataframe(
                    comparison_df,
                    hide_index=True,
                    use_container_width=True,
                )

                fig = go.Figure()

                for run_id, color in [
                    (id_a, "#8E909D"),
                    (id_b, GOLD),
                ]:
                    column = f"Run #{run_id}"

                    fig.add_trace(
                        go.Bar(
                            x=comparison_df["Metric"],
                            y=comparison_df[column],
                            name=column,
                            marker_color=color,
                        )
                    )

                fig.update_layout(
                    barmode="group",
                )

                fig.update_yaxes(
                    title="Score (%)",
                    range=[0, 105],
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True,
                )

                st.divider()

                heading("Execution comparison")

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        f"Run #{id_a} avg latency",
                        f"{df_a['latency'].mean():.2f}s",
                    )
                    st.metric(
                        f"Run #{id_a} tokens",
                        int(df_a["tokens"].sum()),
                    )

                with c2:
                    st.metric(
                        f"Run #{id_b} avg latency",
                        f"{df_b['latency'].mean():.2f}s",
                    )
                    st.metric(
                        f"Run #{id_b} tokens",
                        int(df_b["tokens"].sum()),
                    )

                cases_a = {
                    (item["question"], item["expected"])
                    for item in run_a["results"]
                }

                cases_b = {
                    (item["question"], item["expected"])
                    for item in run_b["results"]
                }

                if cases_a != cases_b:
                    st.warning(
                        "These runs use different test cases. "
                        "Their aggregate scores are not directly "
                        "comparable as a controlled experiment."
                    )
                else:
                    st.info(
                        "Both runs contain the same question "
                        "and reference-answer pairs."
                    )

                st.caption(
                    "Higher similarity does not automatically "
                    "mean better factual accuracy."
                )


# ==============================================
# REPORTS PAGE
# ==============================================

elif st.session_state.page == "Reports":
    page_header(
        "Reports",
        "Evaluation reports",
        "Review and export your latest completed evaluation.",
    )

    latest = get_latest_run()

    if not latest:
        empty_state(
            "No report available",
            "Complete an evaluation to generate a report.",
        )

    else:
        heading("Latest saved results")

        st.caption(
            f"Model: {latest['model']} · "
            f"Run ID: {latest['id'] or 'Not saved'}"
        )

        results_table(latest["results"])

        download_results(
            latest["results"],
            "solvyn_evaluation_report.csv",
            "download_latest_report",
        )

        st.caption(
            "Reports include raw model responses, "
            "reference answers and evaluation metrics."
        )
