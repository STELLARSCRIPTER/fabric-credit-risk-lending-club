# Enterprise Credit Risk Analytics & Default Prediction

**An end-to-end analytics project combining PySpark, XGBoost, and Power BI to analyse loan portfolios, estimate default risk, and support data-driven risk monitoring.**

This project uses historical LendingClub loan data to demonstrate a complete analytics workflow, from data preparation and feature engineering to machine learning evaluation and interactive business intelligence.

The solution combines quantitative analysis, medallion-style data engineering, predictive modelling, and executive-level dashboard design.

## Project Overview

| Component | Details |
|---|---|
| Dataset | LendingClub accepted loan data, 2007–2018 |
| Source data scale | Approximately 2.2 million loan records |
| Scored evaluation dataset | Approximately 269K loans |
| Data processing | PySpark and Parquet |
| Machine learning | XGBoost Classifier |
| Model ROC-AUC | 0.7031 |
| Business intelligence | Power BI Desktop and DAX |
| Report design | Custom JSON themes |
| Architecture | Local implementation of Bronze → Silver → Gold |

## Project Objectives

- Build a structured data pipeline for large-scale historical loan data.
- Clean and transform raw data using PySpark.
- Engineer features relevant to credit default prediction.
- Train and evaluate an XGBoost classification model.
- Analyse portfolio exposure, default patterns, and geographic risk.
- Present model outputs through an interactive Power BI report.
- Explore how predictive risk scores can support portfolio monitoring and prioritisation.

## Architecture

The project follows the **Bronze → Silver → Gold Medallion Architecture pattern**, implemented locally using PySpark and Parquet.

### Bronze — Raw Data Ingestion

- Started with the historical LendingClub dataset.
- Used Spark-compatible data processing and storage workflows.
- Established a raw-data foundation for downstream transformations.

### Silver — Data Cleaning and Preparation

**Notebook:** `02_Silver_Cleaning.ipynb`

- Applied data type conversions and data cleaning using PySpark.
- Prepared relevant loan attributes for analysis and modelling.
- Handled missing values as part of the preparation workflow.
- Excluded post-origination information that could introduce target leakage into model features.

### Gold — Feature Engineering and Machine Learning

**Notebook:** `03_Gold_ML_Model.ipynb`

- Engineered analytical features, including FICO score bands and debt-to-income (DTI) bands.
- Prepared model inputs from relevant borrower and loan characteristics.
- Trained an XGBoost classifier to estimate the probability of default.
- Generated predictions for the evaluation dataset.
- Exported scored loan records for Power BI analysis.

**Implementation note:** The project uses a local PySpark and Parquet implementation inspired by lakehouse architecture. It is not deployed on Microsoft Fabric. The architecture demonstrates transferable data engineering concepts rather than a live Fabric deployment.

## Technology Stack

| Area | Technologies |
|---|---|
| Data engineering | PySpark, Parquet |
| Data preparation | Python, Pandas |
| Machine learning | XGBoost, scikit-learn |
| Data analysis | Python, SQL where applicable |
| Business intelligence | Power BI Desktop, DAX, Power Query |
| Report design | Custom Power BI JSON themes |
| Environment | Python 3.11, Java 17 |

## Power BI Dashboard

The interactive Power BI report contains seven analytical pages.

### 1. Executive Summary

Provides a high-level view of portfolio performance and risk.

- Executive KPIs for loan exposure and observed default rates.
- Predicted-risk and actual-default comparisons.
- FICO-related risk patterns.
- Navigation to detailed analytical views.

### 2. Portfolio Overview

Summarises portfolio composition and key credit-risk indicators.

- Loan portfolio characteristics.
- Exposure and default-rate KPIs.
- Overview of risk distribution and portfolio trends.

### 3. Portfolio Risk Profile

Explores observed default patterns across loan characteristics.

- Default-rate analysis by DTI band.
- Comparison across loan terms.
- Default patterns by loan purpose.
- Identification of segments with comparatively higher observed default rates.

### 4. Geographic Risk Profile

Examines geographic differences in loan exposure and default outcomes.

- State-level loan exposure analysis.
- Geographic visualisation of risk variation across the United States.
- State rankings by exposure and observed default rate.
- Comparison of high-exposure locations with high-default-rate locations.

### 5. ML Predictions

Explores the model's predictions and their relationship with observed outcomes.

- Distribution of predicted default probabilities.
- Comparison of predicted risk with actual default outcomes.
- Risk patterns across selected loan and borrower characteristics.
- Visual exploration of interest rates and predicted default risk.

### 6. Model Performance & Validation

Evaluates the classifier and examines its limitations.

- ROC-AUC and classification metrics.
- Confusion matrix analysis.
- Precision, recall, and F1-score.
- False-negative analysis.
- Discussion of the selected classification threshold and its implications for identifying defaults.

### 7. High-Risk Watchlist

Provides a focused view of loans selected for elevated predicted risk.

- A watchlist of 585 loans with predicted default probabilities above 60%.
- Portfolio share analysis for the selected segment.
- A practical illustration of model-assisted risk monitoring and prioritisation.

The watchlist is an analytical tool, not a definitive determination that an individual loan will default.

## Machine Learning Evaluation

The XGBoost classifier achieved a **ROC-AUC of 0.7031** on the reported evaluation dataset.

ROC-AUC measures the model's ability to rank defaulted loans above non-defaulted loans across classification thresholds. The result indicates moderate discriminatory ability and provides a baseline for further improvement.

The evaluation considers several complementary metrics:

- **ROC-AUC:** Measures ranking performance across thresholds.
- **Precision:** Measures the proportion of loans flagged as defaults that are actual defaults.
- **Recall:** Measures the proportion of actual defaults identified by the classifier.
- **F1-score:** Combines precision and recall into a single metric.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives at the selected threshold.

The threshold-based evaluation uses a classification threshold of 0.50. At this threshold, recall is low, meaning many actual defaults are not identified by the binary classifier.

This highlights an important credit-risk modelling consideration: a model may provide useful risk ranking while still missing a substantial proportion of defaults at a particular threshold.

The project therefore examines classification behaviour and missed defaults rather than relying on ROC-AUC alone.

## Power BI Design & Custom JSON Themes

In addition to data modelling and analytics, the project includes custom report styling.

- Created and customised JSON theme files for Power BI.
- Defined consistent colour palettes and visual styling across report pages.
- Experimented with multiple theme variations to explore different professional dashboard designs.
- Applied theme configurations to improve visual consistency and presentation.
- Combined custom styling with interactive visuals, KPIs, geographic analysis, and report navigation.

Theme files are maintained in the `theme/` directory.

The project demonstrates that dashboard development involves more than building charts: visual consistency and clear communication are also important parts of delivering analytical insights.

## Repository Structure

```text
fabric-credit-risk-lending-club/
├── data/
│   ├── .gitkeep
│   ├── accepted_2007_to_2018Q4.csv.gz
│   ├── accepted_2007_to_2018q4.csv/
│   ├── rejected_2007_to_2018q4.csv/
│   ├── silver_lending_club_cleaned.parquet/
│   └── powerbi_credit_risk_predictions.csv
├── docs/
├── notebooks/
│   ├── 02_Silver_Cleaning.ipynb
│   ├── 03_Gold_ML_Model.ipynb
│   ├── test_notebook.ipynb
│   └── hs_err_pid5020.log
├── powerbi/
│   └── LendingClub_Credit_Risk_Dashboard.pbix
├── theme/
│   ├── ConsultingTheme.json
│   ├── CreditRiskTheme.json
│   ├── Institutional_Intelligence.json
│   ├── Nexora_Corporate.json
│   └── Nexora_Light_Crisp.json
├── requirements.txt
└── .gitignore
```

The structure above reflects the local working directory. Development-only files and generated data outputs should be excluded from the public repository where appropriate.

## How to Run

### Prerequisites

- Python 3.11
- Java 17
- Power BI Desktop
- The LendingClub dataset from an authorised source

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd fabric-credit-risk-lending-club
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Ensure that Java is installed and configured correctly for PySpark.

### 4. Prepare the input data

Obtain the required LendingClub dataset and place it in the location expected by the notebooks.

If you use a different directory, update the relevant notebook paths.

### 5. Run the notebooks

Run the available notebooks in sequence:

1. `notebooks/02_Silver_Cleaning.ipynb`
2. `notebooks/03_Gold_ML_Model.ipynb`

The Silver notebook prepares the cleaned dataset. The Gold notebook performs feature engineering, model training, evaluation, and prediction export.

The current repository does not contain a dedicated Bronze ingestion notebook, so the data ingestion steps should be followed as implemented in the available notebooks.

### 6. Open the Power BI report

Open the report:

`powerbi/LendingClub_Credit_Risk_Dashboard.pbix`

If necessary, update its data-source path to point to `data/powerbi_credit_risk_predictions.csv`.

The existing scored CSV can be used to explore the dashboard without rerunning the complete machine learning pipeline, provided the expected output file is available.

## Data Management & Reproducibility

The project uses historical LendingClub data and generated processing outputs.

Large source datasets, Parquet output directories, and scored CSV files may be excluded from GitHub to reduce repository size. Users should obtain the source data through an authorised channel and generate local outputs by running the notebooks.

The repository's `.gitignore` excludes data files and virtual-environment artefacts. Confirm that large files have not already been committed before publishing the repository.

File paths, package versions, and dataset availability may require environment-specific configuration.

## Professional Context

This is an independent portfolio project developed to deepen practical experience in data engineering, predictive analytics, and business intelligence.

Separately, my professional experience at Frost & Sullivan involves Customer Experience and CRM, including CRM operations, lead management, data analysis, and business reporting.

These experiences complement each other: professional CRM work provides business context, while this independent project demonstrates the application of quantitative methods, machine learning, and interactive analytics to a credit-risk problem.

This project does not use Frost & Sullivan confidential data and is not presented as a company-sponsored project.

## Key Takeaways

This project brings together:

- PySpark-based data cleaning and transformation.
- Medallion-style data engineering.
- Feature engineering and supervised machine learning.
- Credit-risk model evaluation and threshold analysis.
- Portfolio exposure and geographic risk analysis.
- DAX-driven interactive reporting.
- Custom Power BI JSON themes and consistent visual design.
- Translation of model outputs into business-oriented risk insights.

**Focus areas:** Credit Risk Analytics · Data Engineering · Machine Learning · Business Intelligence · Quantitative Analysis

## Disclaimer

This project is intended for educational and analytical purposes. It uses historical LendingClub data and does not constitute financial advice or a production credit decision system. Model predictions are estimates, and observed relationships should not be interpreted as causal effects.
