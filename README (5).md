# A Parallel Hybrid Model for 5G Network Congestion Control Using Machine Learning

A hybrid ML system that predicts congestion in 5G networks by combining **supervised learning (prediction)** and **unsupervised learning (clustering)**. The model analyzes traffic patterns, detects anomalies, and forecasts congestion to improve **QoS**, **network reliability**, and **decision-making**.

---

## 🚀 Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Dataset Description](#dataset-description)  
5. [Model Architecture](#model-architecture)  
6. [Evaluation Metrics](#evaluation-metrics)  
7. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Run the App](#run-the-app)  
8. [Project Structure](#project-structure)  
9. [Results & Visualizations](#results--visualizations)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## 📋 Project Overview

This project introduces a **Parallel Hybrid Machine Learning Model** designed for real-time **5G congestion prediction and traffic behavior clustering**.  
It integrates:

- **Supervised ML models** → Predict upcoming congestion levels  
- **Unsupervised ML models** → Cluster traffic patterns & detect abnormal behavior  
- **Hybrid decision engine** → Merges both outputs for better accuracy  
- **Streamlit dashboard** → Live visualization and model interaction  

The system aims to support telecom operators with **reliable forecasting, anomaly detection, and resource optimization**.

---

## ✨ Features

### 🔹 Supervised Learning (Prediction Models)
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor **(Best performer)**  
- Support Vector Regression (SVR)

### 🔹 Unsupervised Learning (Clustering Models)
- KMeans Clustering  
- DBSCAN  
- HDBSCAN **(Best performer)**  
- Hierarchical Clustering  
- PCA for dimensionality reduction  

### 🔹 Hybrid Model Abilities
- Merges regression output + cluster label  
- Detects abnormal patterns before congestion rises  
- Improves prediction consistency  
- Supports network-level intelligent decisions  

---

## 🛠 Tech Stack

### ML / Backend
- Python  
- NumPy, Pandas  
- Scikit-Learn  
- HDBSCAN, SciPy  
- Matplotlib, Seaborn  

### Frontend / Dashboard
- Streamlit  

---

## 📂 Dataset Description

The dataset contains real 5G network KPI indicators:

| Feature | Description |
|--------|-------------|
| Base Station (BS) | Tower or site ID |
| Energy Consumption | Unit power usage |
| Network Load (%) | Real-time traffic load |
| ESMODE | Energy saving mode flag |
| Transmission Power | dB transmission power |
| Time in Seconds | Performance time slot |
| Year, Month, Day | Time segmentation |

---

## 🧩 Model Architecture

```
      5G Dataset
          │
          ▼
   Data Preprocessing
          │
 ┌──────────────────────┬──────────────────────┐
 │  Supervised Models   │ Unsupervised Models  │
 │  (Prediction)        │ (Clustering)         │
 └──────────────────────┴──────────────────────┘
          │
          ▼
   Hybrid Decision Engine
          │
          ▼
     Streamlit Dashboard
```

---

## 📊 Evaluation Metrics

### Supervised Metrics
| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Goodness-of-fit |
| MAE | Absolute error measure |

### Unsupervised Metrics
| Metric | Description |
|--------|-------------|
| Silhouette Score | Cluster quality measure |
| Cluster Separation | Distance between clusters |
| Noise Ratio | DBSCAN/HDBSCAN noise detection |

---

## 🔧 Getting Started

### Prerequisites
- Python 3.8+  
- Pip  
- Streamlit  

---

### Installation

1. Clone the repo  
```bash
git clone https://github.com/Ankithnagaraj/A-Parallel-Hybrid-Model-For-5G-Network-Congestion-Control-Based-On-Using-Machine-Learning-Approach.git
cd A-Parallel-Hybrid-Model-For-5G-Network-Congestion-Control-Based-On-Using-Machine-Learning-Approach
```

2. Install dependencies  
```bash
pip install -r requirements.txt
```

---

### Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 A-Parallel-Hybrid-Model-For-5G-Network-Congestion-Control-Based-On-Using-Machine-Learning-Approach
├── app.py
├── models/
│   ├── supervised_models.py
│   ├── unsupervised_models.py
│   └── hybrid_engine.py
├── data/
│   └── 5g_dataset.csv
├── notebooks/
│   └── model_training.ipynb
├── visuals/
│   └── charts and evaluation images
├── requirements.txt
└── README.md
```

---

## 📊 Results & Visualizations
- Regression model plots  
- Clustering scatter plots with PCA  
- Hybrid decision boundaries  
- Error distribution graphs  
- Streamlit dashboard screenshots  

(*You can add your images inside `visuals/` folder and display them here.*)

---

## 🤝 Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss.

---

## 📄 License
This project is licensed under **MIT License**.  
You may modify and reuse for academic and research purposes.

