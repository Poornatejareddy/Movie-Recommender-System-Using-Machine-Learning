# 🎬 Movie Recommender System Using Machine Learning

This project is an **end-to-end Movie Recommender System** built using **Python and Machine Learning**.  
It combines **Collaborative Filtering (Truncated SVD)** and **Content-Based Filtering (TF-IDF)** to recommend movies based on user preferences.  
The system can be run **locally** or in **Google Colab**, and includes a **Streamlit web app** for interactive recommendations.

---

## 🌟 Preview

<p align="center">
  <img src="streamlit preview.png" alt="Movie Recommender Preview" width="800">
</p>

*(Preview of the Streamlit movie recommender UI)*

---

## 🚀 Features

- ✅ Free **MovieLens 100k dataset** — no paid APIs required  
- ✅ Combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)**  
- ✅ Supports **Hybrid Recommendations (SVD + TF-IDF)**  
- ✅ Real-time recommendations (no retraining required)  
- ✅ Interactive **Streamlit web interface**  
- ✅ Fully works in **Google Colab** via ngrok  

---

## 🧠 Concept Overview

| Component | Description |
|------------|-------------|
| **Dataset** | [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) |
| **Collaborative Filtering** | Learns hidden user–movie interactions using **Truncated SVD** |
| **Content-Based Filtering** | Uses **TF-IDF** on movie titles and genres |
| **Hybrid Model** | Combines both embeddings for improved recommendations |
| **Interface** | Built using **Streamlit**, supports local and Colab environments |

---

## 🧰 Tech Stack

- **Python 3.10+**
- **NumPy, Pandas, Scikit-learn**
- **TruncatedSVD** (for collaborative filtering)
- **TF-IDF Vectorizer** (for content-based filtering)
- **Streamlit** (for UI)
- **pyngrok** (for Colab web access)
- **Joblib** (for model persistence)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository

```bash
git clone https://github.com/Poornatejareddy/Movie-Recommender-System-Using-Machine-Learning.git
cd movie-recommender-ml
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or, if you're using Google Colab, the notebook will automatically install everything.

---

## 💻 How to Run

### ▶️ Option 1 — Run the Notebook (Model Training + Evaluation)

Run the main Jupyter Notebook file:

```bash
jupyter notebook Movie_Recommender_System_Using_ML.ipynb
```

This will:
- Download the MovieLens dataset
- Train SVD (Collaborative) and TF-IDF (Content-Based) models
- Create hybrid embeddings
- Evaluate Precision@10 metric
- Save models into the `/models` folder

### ▶️ Option 2 — Run the Streamlit App Locally

Once the model training is done, launch the Streamlit web app:

```bash
streamlit run app.py
```

Then open the app in your browser at: `http://localhost:8501/`

### ▶️ Option 3 — Run Streamlit in Google Colab

If you're running in Google Colab, use the following commands:

```bash
!pip install streamlit pyngrok -q
!ngrok authtoken YOUR_NGROK_TOKEN
!streamlit run app.py --server.port 8501 &
```

A public URL will appear — click it to open your web app in a browser. 🎬

---

## 📈 Model Evaluation

The recommender system is evaluated using **Precision@10** based on a leave-one-out validation strategy.

**Example:**

```
Mean Precision@10 (hybrid profile): 0.3127
```

---

## 🎯 Example Output

**Input (Movies Liked by User):**
- Toy Story (1995)
- Pulp Fiction (1994)

**Recommended Movies:**
1. Twelve Monkeys (1995) — score=0.872
2. Usual Suspects, The (1995) — score=0.861
3. Braveheart (1995) — score=0.852
4. Apollo 13 (1995) — score=0.838
5. Heat (1995) — score=0.827

---

## 📂 Project Structure

```
movie-recommender-ml/
│
├── data/
│   └── ml-100k/                 # MovieLens dataset (automatically downloaded)
│
├── models/
│   ├── item_latent_aligned.npy  # Truncated SVD embeddings
│   ├── item_tfidf.npy           # TF-IDF feature matrix
│   └── movie_maps.pkl           # Mappings for movie IDs
│
├── assets/
│   └── movie_recommender_preview.gif  # App preview (optional)
│
├── Movie_Recommender_System_Using_ML.ipynb   # Main ML notebook
├── app.py                                    # Streamlit web app
├── requirements.txt                          # Dependencies
└── README.md                                 # Documentation
```

---

## 📦 requirements.txt

```
pandas
numpy
scipy
scikit-learn
joblib
tqdm
requests
streamlit
pyngrok
```

---

## 🧑‍💻 Author

**Poorna Teja Reddy K**

💼 AI & ML Enthusiast | Explainable AI Researcher  
📧 [pore22csaiml@cmrit.ac.in](mailto:pore22csaiml@cmrit.ac.in)  

---

## 💡 Future Improvements

🔹 Integrate TMDb API for real-time 2025+ movie updates  
🔹 Display movie posters and genres in Streamlit UI  
🔹 Add rating-based personalization  
🔹 Deploy backend via FastAPI or Render/HuggingFace Spaces  

---
