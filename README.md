# 🌌 2025 NASA Space Apps Challenge: A World Away — Hunting for Exoplanets with AI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Powered%20by-Gradio-FF4B4B?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNDAgMjQwIj48cGF0aCBmaWxsPSIjRkY0QjRCIiBkPSJNMjE1LjUgMTA1LjVMMTIwIDUwTDI0LjUgMTA1LjVMNzAgMTMxLjVMMTIwIDE2MC41TDE3MCAxMzEuNUwyMTUuNSAxMDUuNXoiLz48L3N2Zz4=)](https://gradio.app)
[![NASA Space Apps](https://img.shields.io/badge/NASA%20Space%20Apps-2025-0056A3?logo=nasa&logoColor=white)](https://www.spaceappschallenge.org/)



> *“A World Away” — Using machine learning to classify exoplanet candidates from Kepler mission data.*

---

## 🚀 Overview

This project was developed for the **NASA Space Apps Challenge 2025** under the theme “A World Away.” It leverages artificial intelligence to analyze planetary and stellar parameters and classify whether a celestial body is a **Confirmed Exoplanet**, a **Candidate**, or a **False Positive**.

Built with Python and Gradio, this web app provides an intuitive interface for both single predictions and bulk classification of exoplanet data. Whether you're a researcher, educator, or space enthusiast, you can explore how AI helps astronomers sift through cosmic noise to find new worlds.

---

## 📁 Repository Structure

```

D:.
├── app.py                  # Main Gradio application script
├── cumulative_transformed.csv   # Preprocessed training dataset
├── kepler_exoplanet.csv    # Raw Kepler exoplanet dataset (source)
├── README.md               # You are here!
├── requirements.txt        # Python dependencies
└── test_sample.csv         # Sample data for quick testing

````

---

## ⚙️ How to Use

### Option 1: Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2025_NASA_Space_Apps_Challenge_A_world_away_hunting_for_exoplanets_with_AI.git
   cd 2025_NASA_Space_Apps_Challenge_A_world_away_hunting_for_exoplanets_with_AI


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:

   ```bash
   python app.py
   ```

   → Open `http://localhost:7860` in your browser.

---

### Option 2: Use Online via Hugging Face Spaces

No setup required! Try it live here:
🔗 [2025 NASA Space Apps Challenge: A world away hunting for exoplanets with AI](https://huggingface.co/spaces/mibrahimzia/2025_NASA_Space_Apps_ChallengeA_world_away_hunting_for_exoplanets_with_AI)

---

## 🖥️ Interface Features

The app has three main tabs:

### 1. Single Prediction

Enter values manually for:

* Orbital Period (days)
* Transit Duration (hrs)
* Planet Radius (Earth radii)
* Star Temperature (K)
* Star Radius (Solar radii)

Click **Submit** to get an instant prediction with confidence indicator.

### 2. Bulk CSV Prediction

Upload or select sample datasets (like `test_sample.csv`) to classify multiple entries at once. Results are displayed in a preview table and can be downloaded as a `.csv`.

> ✅ Green check = Confirmed Exoplanet
> 🟡 Yellow dot = Candidate
> ❌ Red X = False Positive

### 3. Insights

View model performance metrics and confidence graphs (see below).

---

## 📊 Model Confidence & Performance Graphs

<img width="633" height="470" alt="proba_dist" src="https://github.com/user-attachments/assets/c643e957-f174-484c-ba67-c6832cf72502" />
Most confirmed exoplanets are predicted with high confidence (right side), while non-exoplanets cluster on the low-confidence side (left). This shows the model makes clear, reliable decisions.

<img width="624" height="624" alt="roc_curve" src="https://github.com/user-attachments/assets/deb94db3-b029-4a69-8959-630556899621" />
The high AUC score (0.93) means the model is very good at telling real exoplanets apart from false signals.

Our classifier outputs not just labels but also **confidence scores**, helping users understand how certain the model is about each prediction. The Insights tab displays visualizations such as:

* Class-wise precision/recall
* Confidence score histograms
* Feature importance rankings

---

## 💡 Tip: Don’t Have Formatted Data?

If you don’t have pre-formatted data, you can:

* Download sample files directly from this repo (e.g., `test_sample.csv`)
* Or use the dropdown menu in the app to choose from built-in examples

---

## 🛠️ Built With

* **Python 3.9+**
* **Gradio**
* **Scikit-learn / XGBoost**
* **Pandas, NumPy**
* **Matplotlib / Plotly**

---

## 📜 License

Licensed under the **Apache License 2.0** — see the [LICENSE](./LICENSE) file.

---

## 🤝 Credits & Inspiration

Inspired by NASA’s Kepler Mission and open-source exoplanet datasets.

---

## 📬 Feedback & Contributions

Open an issue or submit a pull request!


> *Project submitted for the 2025 NASA Space Apps Challenge — “A World Away” track.*

````


If you want, I can also clean up formatting spacing or add more badges (stars, forks, PRs, issues, last commit, etc.).
````
