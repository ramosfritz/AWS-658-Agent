# ⚡ AWS Bedrock 6/58 Ultra Lotto Prediction Agent

A state-of-the-art predictive lottery intelligence platform for the **Philippine Charity Sweepstakes Office (PCSO) Ultra Lotto 6/58**, powered by **AWS Bedrock Amazon Nova Pro (`amazon.nova-pro-v1:0`)** and hybrid machine learning ensembles.

---

## 🌟 Key Features

1. **⚡ AWS Bedrock Amazon Nova Pro Integration**:
   - Deep Bayesian pattern learning and in-context training on 160+ real historical PCSO draws.
   - Dynamic weight calibration across recency momentum, historical frequency, overdue gap intervals, and bell-curve sum optimization.
   - Generates 6 unique winning numbers (1–58) with confidence scores, exact sum, odd:even parity, low:high ratio, and deep mathematical reasoning.

2. **🧠 Dual Engine AI & ML Model Training**:
   - **AWS Bedrock Engine**: Calibrates multi-dimensional lottery probability weights and Markov chain transitions.
   - **Classical ML Suite**: Trains Random Forest, Gradient Boosting, Support Vector Machine (SVC), and Multi-Layer Perceptron (Neural Network).
   - Generates Mean Absolute Error (MAE) benchmarks per ball position (D1 to D6).

3. **📊 Historical Analysis & Live PCSO Scraping**:
   - Automatic headless Chrome scraper with anti-bot protection bypass.
   - Fast local cache (`historical_cache.json`) for instant rendering and offline capability.
   - Interactive historical draws table with CSV export.
   - Interactive Plotly chart of the Jackpot Prize growth in Philippine Pesos (₱).

4. **🔥 Hot, Cold & Skip-Interval Analytics**:
   - Identifies hottest, coldest, and most overdue numbers.
   - Visual 1–58 ball frequency histograms and sum bell-curve distribution.

5. **🔮 3-Draw Sequential Advance Predictor**:
   - Autoregressive projection of the next 3 future consecutive draws (Tuesday, Friday, Sunday).

6. **🎲 AI Custom Strategy Lab**:
   - On-demand combinations: Hot Momentum, Overdue Rebound, Bell-Curve Sum (160–185), High-Jackpot Anti-Clustering, and Lucky Quick Pick.

---

## 🚀 How to Run

1. Open PowerShell or Command Prompt in this folder (`C:\Users\DELL\Documents\AWS-658-Prediction-Agent`).
2. Run via the shortcut script:
   ```cmd
   run_app.bat
   ```
   Or directly with Python:
   ```cmd
   .\.venv\Scripts\python -m streamlit run app.py
   ```

---

## 🔐 Configuration (.env)

The application automatically loads:
```env
# AWS Bedrock Configuration
AWS_REGION=us-east-1

# Option A: Explicit AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key

BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
```
