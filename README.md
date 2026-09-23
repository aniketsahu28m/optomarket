<p align="center">
  <img src="logo.jpg" alt="Optomarket logo" width="360">
</p>

# Optomarket

**AI-driven customer segmentation and analysis tool.**

Optomarket takes a CSV of customer transactions, groups customers into segments with K-Means clustering, visualizes how the segments differ, and uses Google Gemini to write a profile and marketing strategy for each one. It is built with Streamlit, so the whole workflow runs in the browser.

## Features

- **Upload and preview** a transaction CSV from the sidebar.
- **Automatic cleaning**: fills missing numeric and categorical values and calculates each customer's age from their date of birth.
- **Clustering**: segments customers into 4 groups by transaction amount, age, fraud flag and city population.
- **Visualizations**: amount-vs-age scatter plot, violin plot of amounts, age distribution histogram and a 2D PCA plot of the clusters.
- **AI recommendations**: Gemini generates a title, analysis, psychological profile, marketing strategy and fraud-prevention advice for each segment.
- **Downloads**: export the segmented data (CSV) and segment profiles (JSON).

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/aniketsahu28m/optomarket.git
cd optomarket
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey), then set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

The app still runs without a key; only the AI recommendations are disabled.

### 3. Run

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Usage

1. Upload a CSV in the sidebar. A sample dataset, `Augmented_IndiaTransactMultiFacet2024.csv`, is included in this repo.
2. Click **Process Data** to clean the data and run clustering.
3. Use the buttons to switch between **Show Graphs**, **Show Recommendations** and **View Segmented Data**.
4. Download the results with the download buttons.

### Expected CSV columns

Your file needs these columns (missing values are filled in automatically):

`trans_date_trans_time`, `dob`, `amt`, `is_fraud`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`, `customer_id`, `category`, `gender`, `city`, `state`, `job`, `first`, `last`, `merchant`

The included sample dataset is synthetic; it contains no real customer or card data.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repo with `app.py` as the main file.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Google Gemini API

## Troubleshooting

- **"No Gemini API key detected"**: set `GEMINI_API_KEY` before starting the app (or in Streamlit Cloud secrets).
- **"Gemini did not return a result"**: check the key is valid and that you have not run out of API quota.
- **Recommendations take a while**: Gemini writes a long, detailed report for all four segments, which usually takes 20–60 seconds.
