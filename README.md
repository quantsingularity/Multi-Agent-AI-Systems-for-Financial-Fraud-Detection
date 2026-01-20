# Multi-Agent AI Systems for Financial Fraud Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](code/requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This repository provides a complete, reproducible implementation of a **Multi-Agent AI System** for real-time financial fraud detection. The system integrates traditional machine learning anomaly detectors with advanced LLM-powered agents to enhance detection accuracy, provide human-readable explanations, and enforce strict privacy and compliance safeguards.

### Key Features

| Feature                   | Description                                                                                                                                                           |
| :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hybrid Detection**      | Combines ensemble ML models (XGBoost, Isolation Forest) for high-speed scoring with LLM agents for complex case analysis.                                             |
| **Explainability**        | **Narrative Generator Agent** creates human-readable case reports and rationales for flagged transactions, addressing regulatory "Right to Explanation" requirements. |
| **Privacy-First Design**  | **Privacy Guard Agent** ensures PII redaction, rate limiting, and policy enforcement before data is processed by LLMs.                                                |
| **Real-Time Performance** | Optimized for low-latency detection, with a mean processing time of **~127ms** per transaction.                                                                       |
| **Full Reproducibility**  | Deterministic synthetic data generation and a Docker environment ensure all experimental results can be replicated.                                                   |

## 📊 Key Results (Synthetic Data Evaluation)

The full multi-agent system significantly improves performance, particularly in precision, by leveraging the LLM agent to validate and contextualize alerts.

| Model                       | Precision | Recall | F1 Score | AUC-ROC  |
| :-------------------------- | :-------- | :----- | :------- | :------- |
| Isolation Forest            | 0.42      | 0.68   | 0.52     | 0.84     |
| XGBoost                     | 0.71      | 0.83   | 0.76     | 0.92     |
| Ensemble Detector           | 0.74      | 0.81   | 0.77     | 0.93     |
| **Full Multi-Agent System** | **0.78**  | 0.79   | **0.78** | **0.94** |

**Performance Metrics:**

- **Mean Detection Latency**: 127ms
- **P95 Detection Latency**: 340ms
- **Investigator Time Saved**: Estimated **67% reduction** in false positive review time due to high-quality narratives.

## 🚀 Quick Start

The recommended approach is to use Docker for a fully isolated and reproducible environment.

### Prerequisites

- Docker (version 20.10+)
- Python 3.9+ (for local run)

### Run with Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/Multi-Agent-AI-Systems-for-Financial-Fraud-Detection
cd Multi-Agent-AI-Systems-for-Financial-Fraud-Detection

# Build container image
docker build -t fraud-detection-agents .

# Run the full experiment pipeline (generates data, trains models, runs evaluation)
# Results will be saved to the 'results' and 'figures' directories in your host machine.
docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/figures:/app/figures fraud-detection-agents python code/scripts/run_experiment.py --mode full
```

### Local Python Environment

```bash
# Clone and navigate to the code directory
git clone https://github.com/quantsingularity/Multi-Agent-AI-Systems-for-Financial-Fraud-Detection
cd Multi-Agent-AI-Systems-for-Financial-Fraud-Detection/code

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full experiment
python scripts/run_experiment.py --mode full
```

## 📁 Repository Structure

The project is organized into modular components for agents, models, data, and orchestration.

```
Multi-Agent-AI-Systems-for-Financial-Fraud-Detection/
├── README.md                          # This file
├── LICENSE                            # Project license
├── Dockerfile                         # Defines the reproducible environment
│
├── code/                              # Main implementation source code
│   ├── agents/                        # LLM-powered agents (Narrative, Privacy Guard)
│   │   ├── llm_agents.py             # Evidence Aggregator & Narrative Generator
│   │   └── privacy_guard.py          # PII redaction and policy enforcement
│   │
│   ├── models/                        # ML-based anomaly detectors
│   │   └── anomaly_detectors.py      # Isolation Forest, XGBoost, Ensemble
│   │
│   ├── orchestrator/                  # Central coordination logic
│   │   └── orchestrator.py           # The main FraudDetectionOrchestrator class
│   │
│   ├── data/                          # Data generation and feature engineering
│   │   ├── synthetic_generator.py    # Deterministic data generation
│   │   └── feature_engineering.py    # Feature engineering pipeline
│   │
│   ├── scripts/                       # Experiment runners
│   │   └── run_experiment.py         # Main script for quick and full runs
│   │
│   └── config.py                      # Configuration settings
│
├── figures/                           # Generated publication-ready plots
└── results/                           # Experiment outputs (metrics, logs)
```

## 🏗️ System Architecture

The system is built around the **FraudDetectionOrchestrator**, which coordinates the workflow across specialized, modular components.

### Agent Hierarchy and Workflow

| Conceptual Agent        | Responsibility                                                                             | Implementation Location             |
| :---------------------- | :----------------------------------------------------------------------------------------- | :---------------------------------- |
| **Orchestrator**        | Manages the end-to-end workflow, from feature engineering to final case report generation. | `code/orchestrator/orchestrator.py` |
| **Feature Engineer**    | Extracts temporal, statistical, and behavioral features from raw transaction data.         | `code/data/feature_engineering.py`  |
| **Anomaly Detector**    | Runs the ensemble of ML models (XGBoost, Isolation Forest) to assign a raw fraud score.    | `code/models/anomaly_detectors.py`  |
| **Privacy Guard**       | Redacts PII (e.g., card numbers) and enforces rate limiting/policy gates.                  | `code/agents/privacy_guard.py`      |
| **Evidence Aggregator** | Consolidates scores, features, and policy violations into a structured evidence package.   | `code/agents/llm_agents.py`         |
| **Narrative Generator** | Uses the evidence package to generate a human-readable case report for investigators.      | `code/agents/llm_agents.py`         |

## 🛡️ Privacy and Compliance

The system is designed with a strong focus on regulatory compliance and data privacy, integrating safeguards directly into the agent workflow.

| Safeguard                | Description                                                                                                  | Compliance Benefit                                                       |
| :----------------------- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **PII Redaction**        | Automatic, policy-driven redaction of sensitive data (e.g., card numbers, SSN) before LLM processing.        | Prevents data leakage and aids GDPR/CCPA compliance.                     |
| **Audit Logs**           | Complete, timestamped trace of all agent decisions, scores, and policy checks for every flagged transaction. | Supports regulatory auditability and internal review.                    |
| **Rate Limiting**        | Policy enforcement (e.g., max 10 flags per user per day) to prevent alert fatigue and false positive spikes. | Improves operational efficiency and investigator focus.                  |
| **Narrative Generation** | Provides a clear, evidence-backed rationale for why a transaction was flagged.                               | Addresses the "Right to Explanation" under GDPR and similar regulations. |

## 🔬 Reproducing Results

The experiments are designed to be fully deterministic using a fixed random seed (42) for data generation and model training.

### Full Experiment Pipeline

```bash
# 1. Generate synthetic data (deterministic)
python code/data/synthetic_generator.py

# 2. Train models and run evaluation
python code/scripts/run_experiment.py --mode full

# 3. Generate figures from results
python code/eval/generate_figures.py

# View results
cat results/metrics/baseline_metrics.json
ls figures/
```
