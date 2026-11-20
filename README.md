# ◆ SOLVYN

**An AI Evaluation Workspace for measuring, inspecting, and comparing language model responses.**

SOLVYN is a Python-based evaluation application that helps developers run repeatable LLM tests, inspect model outputs, track execution performance, and compare evaluation results across experiments.

Built with Python, Streamlit, Groq, Sentence Transformers, Plotly, and SQLite.

## Features

- **LLM Evaluation:** Execute question-and-answer datasets against a Groq-hosted language model.
- **Exact Match:** Compare normalized reference and generated answers.
- **Token F1:** Measure word-level precision and recall.
- **Keyword Coverage:** Measure the overlap of unique reference-answer words.
- **Semantic Similarity:** Compare sentence embeddings using `all-MiniLM-L6-v2` and cosine similarity.
- **Performance Analytics:** Visualize evaluation metrics, response latency, and token consumption.
- **Custom Datasets:** Upload, edit, and validate CSV evaluation datasets.
- **Run History:** Save and retrieve completed evaluation runs using SQLite.
- **Run Comparison:** Compare two experiments, including score differences and execution statistics.
- **CSV Reporting:** Export detailed evaluation results.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and evaluation logic |
| Streamlit | Interactive user interface |
| Groq API | Language model inference |
| Sentence Transformers | Semantic embeddings |
| Pandas | Data processing |
| Plotly | Interactive analytics |
| SQLite | Local evaluation history |
| unittest | Automated tests |

## How It Works

```text
Evaluation Dataset
        |
        v
    Groq Model
        |
        v
  Generated Answer
        |
        v
  Evaluation Engine
        |
        +-- Exact Match
        +-- Token F1
        +-- Keyword Coverage
        +-- Semantic Similarity
        |
        v
 SQLite Run History
        |
        v
 Analytics / Comparison / CSV
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Tarikh792/Solvyn.git
cd Solvyn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Your API Key

Copy `.env.example` to `.env` and provide your own Groq API key:

```dotenv
GROQ_API_KEY=your_actual_key_here
```

Never commit your `.env` file.

### 5. Launch SOLVYN

```bash
python -m streamlit run app.py
```

Open the local address displayed by Streamlit.

## Evaluation Datasets

SOLVYN accepts CSV datasets with these required columns:

| question | expected |
|---|---|
| What is the capital of France? | Paris |
| What does LLM stand for? | Large Language Model |

Each dataset can contain up to 100 valid test cases.

## Running Tests

```bash
python -m unittest test_solvyn.py -v
```

The tests cover evaluation metrics and basic database persistence.

## Evaluation Methodology

SOLVYN combines deterministic lexical metrics with embedding-based semantic similarity.

**Important limitations:**

- Exact Match is sensitive to differences in wording after normalization.
- Token F1 and Keyword Coverage measure word overlap, not factual truth.
- Semantic Similarity measures embedding similarity, not verified correctness.
- Higher scores do not guarantee that a response is factually accurate.
- Comparing runs with different datasets may produce misleading conclusions.

SOLVYN is intended as a practical evaluation workbench, not a certified hallucination detector.

## Storage and Deployment

Evaluation history is stored in a local SQLite database.

The development version is intended for local execution. Deploying to an ephemeral cloud environment requires a persistent external database for reliable history retention.

Datasets currently remain in the active Streamlit session.

## Project Structure

```text
Solvyn/
├── app.py
├── evaluator.py
├── database.py
├── test_solvyn.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Future Improvements

- Persistent cloud database
- Multiple model providers
- Human-reviewed correctness evaluation
- Groundedness evaluation against supplied evidence
- Model cost estimation
- Larger automated regression suites

## 🎬 Project Demo

Watch SOLVYN in action — AI evaluations, semantic similarity,
performance analytics, run comparisons, and reporting.

[▶ Watch SOLVYN Portfolio Demo](./SOLVYN%20Portfolio%20Demo.mp4)

## Author

**Syed Ali Tarikh**

GitHub: [@Tarikh792](https://github.com/Tarikh792)

Built as part of a practical AI Engineering portfolio.
