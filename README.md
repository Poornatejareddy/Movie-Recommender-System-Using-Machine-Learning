# 🎬 Movie Recommender System Using Machine Learning

This project is an **end-to-end Movie Recommender System** built using **Python and Machine Learning**.  
It combines **Collaborative Filtering (Truncated SVD)** and **Content-Based Filtering (TF-IDF)** to recommend movies based on user preferences.  
The system can be run **locally** or in **Google Colab**, and includes a **Streamlit web app** for interactive recommendations.

---

## 🌟 Preview

<p align="center">
  <img src="assets/movie_recommender_preview.gif" alt="Movie Recommender Preview" width="800">
</p>

*(Preview of the Streamlit movie recommender UI)*

---

## 🚀 Features

- ✅ Free **MovieLens 100k dataset** — no paid APIs
- ✅ Combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)**
- ✅ Supports **Hybrid Recommendations (SVD + TF-IDF)**
- ✅ Real-time user profile updates (no retraining needed)
- ✅ Interactive **Streamlit UI**
- ✅ Works in **Google Colab** via ngrok

---

## 🧠 Concept Overview

| Component | Description |
|------------|-------------|
| **Dataset** | [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) |
| **Collaborative Filtering** | Learns hidden user–movie patterns using **Truncated SVD** |
| **Content-Based Filtering** | Uses **TF-IDF** to analyze movie titles and genres |
| **Hybrid Model** | Combines both representations for better accuracy |
| **Interface** | Streamlit app (local or Colab-compatible) |

---

## 🧰 Tech Stack

- **Python 3.10+**
- **NumPy, Pandas, Scikit-learn**
- **TruncatedSVD** (Collaborative)
- **TF-IDF Vectorizer** (Content-based)
- **Streamlit** (UI)
- **pyngrok** (Colab web access)

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/movie-recommender-ml.git
cd movie-recommender-ml
