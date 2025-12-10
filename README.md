# A-Parallel-Hybrid-Model-For-5G-Network-Congestion-Control-Based-On-Using-Machine-Learning-Approach
A hybrid machine learning system that predicts network congestion in 5G environments using combined supervised and unsupervised learning. The project analyzes traffic patterns, detects anomalies, and forecasts congestion to help improve QoS, network reliability, and decision-making.

**🧠 Key Features**

**🔹 Supervised Learning (Prediction)**

Linear Regression
Decision Tree Regressor
Random Forest Regressor (Best performer)
Support Vector Machine (SVM)

**🔹 Unsupervised Learning (Clustering)**

KMeans
DBSCAN
HDBSCAN (Best performer)
Hierarchical Clustering
PCA visualization

**🔹 Hybrid Model Capabilities**

Combines predictions & clustering
Enhances congestion prediction reliability
Helps understand traffic behavior patterns

**🧩 System Architecture**
**5G Dataset
   │
   ▼
Preprocessing
   │
 ┌───────────────┬─────────────────┐
 │ Supervised ML │ Unsupervised ML │
 │ (Prediction)   │ (Clustering)    │
 └───────────────┴─────────────────┘
   │
   ▼
Hybrid Decision Engine
   │
   ▼
Streamlit Dashboard**

**🗂 Dataset Description**
Dataset includes 5G performance indicators:
1.Base Station (BS)
2.Energy consumption
3.Network Load (%)
4.ESMODE (Energy saving mode flag)
5.Transmission Power
6.Time in Seconds
7.Year, Month, Day

**📊 Evaluation Metrics**
**Supervised Learning**
**Metric	**            **Purpose**
MSE	                Measures average squared error
RMSE	              Square root of MSE
R² Score	          Goodness of fit
MAE	                Average absolute error

**Unsupervised Learning**
**Metric**	               ** Purpose**
Silhouette Score	        Cluster quality
Cluster Separation	      Distance between clusters
Noise Ratio	              Useful for DBSCAN/HDBSCAN


**🖥 Technologies Used**

1.Python
2.NumPy, Pandas
3.Scikit-Learn
4.HDBSCAN, SciPy
5.Matplotlib, Seaborn
6.Streamlit

**How to Run**
**1️⃣ Clone the Repository**
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

**2️⃣ Install Dependencies**
pip install -r requirements.txt

**3️⃣ Launch Dashboard**
streamlit run app.py

**📁 Project Structure**

/5G-Hybrid-Model
│── app.py
│── model_supervised.py
│── model_unsupervised.py
│── hybrid_engine.py
│── requirements.txt
│── dataset.csv
│── README.md
│── images/
      ├── banner.png
      ├── flowchart.png
