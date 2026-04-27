# Pneumonia Detection with AI Governance

This repository contains a deep learning project for detecting pneumonia from chest X-ray images. Beyond standard model training, this project integrates AI governance principles, including fairness auditing, robustness testing, and human-in-the-loop (HITL) oversight.

## Features

- **Pneumonia Detection Model**: A ResNet18-based architecture trained to classify chest X-rays.
- **Web Dashboard**: A Flask-based web interface (`app.py`) where clinicians can upload X-rays for prediction.
- **Fairness Auditing**: Tools to detect and mitigate bias in the model's predictions across different patient demographic groups (`fairness_auditor.py`).
- **Robustness Testing**: Evaluation scripts to test how well the model handles noisy or adversarial inputs (`robustness_tester.py`).
- **Human-in-the-Loop Logging**: A secure logging mechanism in the dashboard that records when a doctor approves, overrides, or reviews a model's prediction.
- **Privacy Shield**: An optional feature during prediction to handle sensitive patient data securely.

## Project Structure

- `app.py`: The main Flask application providing the web dashboard and API endpoints.
- `model.py` / `train.py`: Scripts for defining and training the ResNet18 model.
- `data_loader.py`: Handles loading and preprocessing the chest X-ray images.
- `utils.py`: Helper functions for model inference and metric extraction.
- `config.yaml`: Central configuration file for model, training, and data parameters.
- `fairness_auditor.py`: Scripts for calculating metrics like demographic parity difference.
- `robustness_tester.py`: Evaluates model stability and reliability.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/usmanwajid09/Pneumonia-Detection.git
   cd Pneumonia-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web dashboard:**
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.

## Group Members

- Usman Wajid
- Hamza Hussain
- Mahhee Ibn Ahmer
- Muhammad Faizan
