# Multi-Agent AI Systems for Financial Fraud Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](code/requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This repository provides a complete, reproducible implementation of a **Multi-Agent AI System** for real-time financial fraud detection. The system is designed to overcome the limitations of traditional single-model systems by integrating high-speed machine learning anomaly detectors with advanced LLM-powered agents. This hybrid approach significantly enhances detection accuracy, provides human-readable explanations for every alert, and enforces strict privacy and compliance safeguards.

The core of the system is the `FraudDetectionOrchestrator`, which manages a workflow that includes PII redaction, ensemble scoring, evidence aggregation, and narrative generation for suspicious transactions.

---

## 🔑 Key Features and Capabilities

The system is engineered for production deployment, focusing on low-latency performance, explainability, and compliance.

| Feature                           | Category       | Key Capabilities                                                                                                                                              |
| :-------------------------------- | :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Hybrid Detection Architecture** | Core Logic     | Combines ensemble ML models (**XGBoost, Isolation Forest**) for high-speed scoring with **LLM Agents** for complex case analysis and narrative generation.    |
| **Privacy-First Design**          | Compliance     | **Privacy Guard Agent** enforces PII redaction, rate limiting, and policy checks before data is exposed to LLMs or investigators.                             |
| **Explainability & Audit**        | Regulatory     | **Narrative Generator Agent** creates human-readable case reports and rationales for flagged transactions, addressing the "Right to Explanation" requirement. |
| **Real-Time Performance**         | Scalability    | Optimized for low-latency detection, achieving a mean processing time of **~127ms** per transaction, suitable for high-volume environments.                   |
| **Cost-Benefit Analysis**         | Business Logic | Utility module for quantifying the financial impact of False Positives (FP) and False Negatives (FN) to optimize the alert threshold.                         |
| **Production Deployment Ready**   | Infrastructure | Includes **Kubernetes (k8s)** manifests and **Helm charts** for seamless, scalable deployment in a cloud environment.                                         |

---

## 🤖 Multi-Agent Architecture: The Fraud Detection Workflow

The system's workflow is managed by the `FraudDetectionOrchestrator` (`code/orchestrator/orchestrator.py`), which coordinates the following specialized components:

| Component               | Agent Type | Responsibility                                                                                        | Implementation Location            |
| :---------------------- | :--------- | :---------------------------------------------------------------------------------------------------- | :--------------------------------- |
| **Feature Engineer**    | Utility    | Extracts temporal, statistical, and behavioral features from raw transaction data.                    | `code/data/feature_engineering.py` |
| **Privacy Guard**       | Agent      | Redacts PII (e.g., card numbers) and enforces policy gates (e.g., rate limiting).                     | `code/agents/privacy_guard.py`     |
| **Anomaly Detector**    | Model      | Runs the ensemble of ML models to assign a raw fraud score (Probability of Fraud).                    | `code/models/anomaly_detectors.py` |
| **Evidence Aggregator** | LLM Agent  | Consolidates ML scores, feature importance, and policy violations into a structured evidence package. | `code/agents/llm_agents.py`        |
| **Narrative Generator** | LLM Agent  | Uses the evidence package to generate a concise, human-readable case report for investigators.        | `code/agents/llm_agents.py`        |
| **Online Learning**     | Utility    | Supports incremental model updates and adaptation to new fraud patterns.                              | `code/utils/online_learning.py`    |

---

## 📁 Repository Structure and Component Breakdown

The project is organized into modular components for agents, models, data, and infrastructure.

### Top-Level Structure

| Path       | Description                                                                             |
| :--------- | :-------------------------------------------------------------------------------------- |
| `code/`    | Contains all Python source code for the agents, models, and system components.          |
| `docs/`    | Contains supplementary documentation, including `PRODUCTION_DEPLOYMENT.md`.             |
| `figures/` | Stores generated plots and visualizations (e.g., model comparison, latency).            |
| `k8s/`     | Kubernetes manifests for deploying the system components (e.g., deployments, services). |
| `helm/`    | Helm chart for simplified, parameterized deployment of the entire system.               |

### Detailed `code/` Directory Breakdown

| Directory            | Key File(s)                                        | Detailed Function                                                                   |
| :------------------- | :------------------------------------------------- | :---------------------------------------------------------------------------------- |
| `code/orchestrator/` | `orchestrator.py`                                  | The main class coordinating the entire fraud detection workflow.                    |
| `code/agents/`       | `llm_agents.py`, `privacy_guard.py`                | Implementation of the LLM-powered agents and the PII protection layer.              |
| `code/models/`       | `anomaly_detectors.py`                             | Contains the Isolation Forest, XGBoost, and Ensemble model implementations.         |
| `code/data/`         | `synthetic_generator.py`, `feature_engineering.py` | Modules for deterministic data generation and feature extraction.                   |
| `code/utils/`        | `cost_benefit_analysis.py`, `online_learning.py`   | Utility functions for financial analysis, imbalance handling, and model adaptation. |
| `code/eval/`         | `generate_figures.py`                              | Scripts for running the evaluation and generating publication-ready plots.          |
| `code/scripts/`      | `run_experiment.py`                                | Main entry point for running the quick and full experimental pipelines.             |

---

## 📈 Key Results and Performance Benchmarks

The multi-agent system demonstrates superior performance, particularly in precision, by leveraging the LLM agent to validate and contextualize alerts, leading to a reduction in False Positives.

| Model                       | Precision | Recall | F1 Score | AUC-ROC  |
| :-------------------------- | :-------- | :----- | :------- | :------- |
| Isolation Forest            | 0.42      | 0.68   | 0.52     | 0.84     |
| XGBoost                     | 0.71      | 0.83   | 0.76     | 0.92     |
| Ensemble Detector           | 0.74      | 0.81   | 0.77     | 0.93     |
| **Full Multi-Agent System** | **0.78**  | 0.79   | **0.78** | **0.94** |

**Operational Performance:**

| Metric                      | Value                       | Implication                                                                       |
| :-------------------------- | :-------------------------- | :-------------------------------------------------------------------------------- |
| **Mean Detection Latency**  | 127ms                       | Suitable for real-time transaction processing.                                    |
| **P95 Detection Latency**   | 340ms                       | Ensures consistent, low-latency performance under load.                           |
| **Investigator Time Saved** | Estimated **67% reduction** | High-quality narratives drastically reduce the time spent reviewing false alerts. |

---

## 🚀 Quick Start

The recommended approach is to use Docker for a fully isolated and reproducible environment.

### Prerequisites

- Docker (version 20.10+)
- Python 3.9+ (for local run)

### Run with Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/quantsingularity/Multi-Agent-AI-Systems-for-Financial-Fraud-Detection.git
cd Multi-Agent-AI-Systems-for-Financial-Fraud-Detection

# 2. Build container image
docker build -t fraud-detection-agents .

# 3. Run the full experiment pipeline
# Results will be saved to the 'results' and 'figures' directories in your host machine.
docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/figures:/app/figures fraud-detection-agents python code/scripts/run_experiment.py --mode full
```

---

## 🛡️ Privacy and Compliance

The system's design incorporates several layers of compliance and privacy protection, primarily managed by the **Privacy Guard Agent**.

| Safeguard                | Description                                                                    | Compliance Benefit                                                       |
| :----------------------- | :----------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **PII Redaction**        | Automatic, policy-driven redaction of sensitive data before LLM processing.    | Prevents data leakage and aids GDPR/CCPA compliance.                     |
| **Audit Logs**           | Complete, timestamped trace of all agent decisions and scores.                 | Supports regulatory auditability and internal review.                    |
| **Narrative Generation** | Provides a clear, evidence-backed rationale for why a transaction was flagged. | Addresses the "Right to Explanation" under GDPR and similar regulations. |

---

## 📄 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.
