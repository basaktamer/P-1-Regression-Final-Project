# 🏠 Adana Rental Price Prediction
**End-to-End Machine Learning & Deep Learning Project**

## 📌 Project Overview
This project predicts apartment rental prices in **Adana, Turkey**. It features a complete pipeline: from **web scraping** 1,464 listings to deploying a **Deep Learning inference engine** via Streamlit.

---

## 🛠️ Tech Stack & Workflow
* **Data Collection:** Selenium (Undetected Chromedriver).
* **Preprocessing:** Log Transformation (`np.log1p`), One-Hot Encoding, and `StandardScaler`.
* **Modeling:** Ridge Regression, CatBoost, and **Keras (TensorFlow)**.
* **Deployment:** Streamlit & Scikit-Learn Pipelines.

---

## 📊 Model Performance ($R^2$ Score)
| Model | Algorithm | R-Squared |
| :--- | :--- | :--- |
| **Baseline** | Ridge Regression | **0.3721** |
| **Boosting** | CatBoost | 0.3375 |
| **Deep Learning** | Keras (MLP) | **0.2276** |

> **Analysis:** While the Neural Network used **BatchNormalization** and **Dropout**, the Ridge model proved more robust for this specific tabular dataset size ($N \approx 1400$).

---

## 🧠 Deep Learning Architecture
The Keras model was designed with a "block" structure:
1.  **Input Layer:** 128 neurons + BatchNormalization.
2.  **Hidden Layer:** 64 neurons + Dropout (0.3).
3.  **Refinement:** 32 neurons.
4.  **Output:** 1 neuron (Linear activation for regression).

---

## 🚀 How to Run
1. **Install Dependencies:**
   `pip install -r requirements.txt`
2. **Launch the App:**
   `streamlit run app.py`

---

## 📂 Project Structure
* `notebook.ipynb`: EDA and Model Training.
* `app.py`: Streamlit User Interface.
* `adana_rent_pipeline.pkl`: Serialized model and scaler.
* `requirements.txt`: Environment dependencies.

---
**Developed by Basak | March 2026**