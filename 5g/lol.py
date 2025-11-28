import numpy as np
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from logic.supervised import train_regression
from logic.unsupervised import run_kmeans, run_pca
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Machine Learning Dashboard", layout="wide")
st.title("📊 Machine Learning Dashboard: Hybrid Model Approach ")

option = st.sidebar.selectbox("Select Mode", ["Supervised Learning", "Unsupervised Learning"])

default_dataset = "small.csv" if option == "Unsupervised Learning" else "supervised.csv"
all_datasets = os.listdir("datasets")
if default_dataset not in all_datasets:
    default_dataset = all_datasets[0]
file = st.sidebar.selectbox("Select Dataset", all_datasets, index=all_datasets.index(default_dataset))

if st.sidebar.button("📄 REPORT"):
    st.subheader("📝 Full Project Report")
    st.markdown("""
    ### Project Title:
    Congestion Control Prediction Model for 5G Environment Based on Supervised and Unsupervised Machine Learning Approach

    ### Objective:
    Predict congestion in a 5G environment using regression and clustering techniques, leveraging both supervised and unsupervised algorithms for improved accuracy and traffic classification.
    """)

    report_content = """Project Title: 5G Congestion Control Using ML\n\nSupervised Models:\n- Random Forest (R2: 0.96)\n- Decision Tree (R2: 0.93)\n- Linear Regression (R2: 0.55)\n- SVM (R2: -0.02)\n\nUnsupervised Models:\n- HDBSCAN (~71%)\n- KMeans\n- Agglomerative\n\nHybrid Approach: Parallel\nOutcome: Best models are Random Forest & HDBSCAN\n"""
    st.download_button("⬇️ Download Report", report_content, file_name="5G_Project_Report.txt")

if file:
    df = pd.read_csv(f"datasets/{file}")

    st.markdown("This dashboard demonstrates supervised (regression) and unsupervised (clustering and PCA) machine learning techniques.")

    # Feature Distribution Overview
    st.subheader("📈 Feature Distribution Overview")
    numeric_cols = df.select_dtypes(include=['number']).columns
    selected_feature = st.selectbox("Choose a numeric feature to view its distribution:", numeric_cols)
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')  # Reduced size
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax, color="skyblue")
    ax.set_title(f"Distribution of {selected_feature}", color='black')
    ax.set_xlabel(selected_feature, color='black')
    ax.set_ylabel("Frequency", color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    st.pyplot(fig)
    st.markdown(
        f"- This histogram shows the distribution of {selected_feature} values in the selected dataset.\n"
        f"- Most values are concentrated around the mean, revealing how data is spread for model input."
    )

    if option == "Supervised Learning":
        st.subheader("🤖 Supervised Regression Analysis")
        target = st.selectbox("🎯 Select Target Column", df.columns)
        st.markdown("### 🧪 Enter Input Values for Prediction")
        input_data = {}
        for col in df.columns:
            if col != target and np.issubdtype(df[col].dtype, np.number):
                val = st.number_input(f"{col}", value=float(df[col].mean()))
                input_data[col] = val

        if st.button("🔁 Run Regression"):
            result = train_regression(df, target)
            y_test = np.array(result['y_test'])
            y_pred = np.array(result['predictions'])

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.markdown(f"**📉 MSE:** {mse:.3f}")
            st.markdown(f"**📊 RMSE:** {rmse:.3f}")
            st.markdown(f"**📈 R² Score:** {r2:.3f}")
            st.markdown(f"**🧮 MAE:** {mae:.3f}")

            st.subheader("📍 Actual vs Predicted")
            chart_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.line_chart(chart_df, use_container_width=False)
            st.markdown(
                "- The line chart compares actual vs. predicted target values for regression.\n"
                "- Good alignment means the model predicts accurately; discrepancies indicate prediction errors."
            )

            st.subheader("📌 Prediction Error Plot")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="purple")
            ax.set_xlabel("Actual", color='black')
            ax.set_ylabel("Predicted", color='black')
            ax.set_title("Actual vs Predicted Scatter Plot", color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            st.pyplot(fig)
            st.markdown(
                "- Each point represents a test sample's actual and predicted value.\n"
                "- Points near the diagonal show better predictions; more scatter shows higher errors."
            )

            user_df = pd.DataFrame([input_data])
            for col in result['columns']:
                if col not in user_df.columns:
                    user_df[col] = 0
            user_df = user_df[result['columns']]
            user_pred = result['model'].predict(user_df)
            st.success(f"🔮 Predicted {target}: {user_pred[0]:.2f}")

        st.subheader("✅ Best Performing Model Summary")
        st.markdown(
            "**Random Forest** is the most accurate supervised regression model in this project with an R² score of **0.96**, followed by Decision Tree (**0.93**) and Linear Regression (**0.55**). SVM performed poorly."
        )

    elif option == "Unsupervised Learning":
        st.subheader("🧠 Unsupervised Learning")
        method = st.radio("Choose Method", ["KMeans Clustering", "PCA"])

        if method == "KMeans Clustering":
            k = st.slider("🔢 Number of Clusters", 2, 10, 3)
            clustered = run_kmeans(df.select_dtypes(include='number'), n_clusters=k)
            st.write("🔍 Clustered Data")
            st.dataframe(clustered)

            st.subheader("📍 Cluster Plot")
            if 'PCA1' in clustered.columns and 'PCA2' in clustered.columns:
                fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
                sns.scatterplot(data=clustered, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax)
                ax.set_title("KMeans Clustering (PCA Projection)", color='black')
                ax.set_xlabel("PCA1", color='black')
                ax.set_ylabel("PCA2", color='black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                st.pyplot(fig)
                st.markdown(
                    "- This plot shows clusters identified using KMeans projected onto two principal components.\n"
                    "- Well-separated clusters suggest good differentiation of network traffic patterns."
                )

            st.subheader("✅ Clustering Accuracy Summary")
            st.markdown(
                "Based on evaluation reports, **HDBSCAN** achieved the highest clustering accuracy (~71%), making it the most effective unsupervised model in this project."
            )

        elif method == "PCA":
            comp = st.slider("⚙️ PCA Components", 2, min(5, len(df.columns)-1), 2)
            pca_df, variance_ratio = run_pca(df.select_dtypes(include='number'), n_components=comp)
            st.write("📈 PCA Output")
            st.dataframe(pca_df)

            st.subheader("📊 PCA Component Trend")
            st.line_chart(pca_df, use_container_width=False)
            st.markdown(
                "- This line chart displays the principal component scores for each sample.\n"
                "- Higher variance in components means they capture more useful information from the data."
            )

            st.subheader("🧪 Explained Variance Ratio")
            st.write(variance_ratio)
            st.markdown(
                "- The variance ratio shows how much total variance is captured by each PCA component.\n"
                "- Components explaining higher variance are more important for representing the dataset."
            )

            if comp <= 3:
                st.subheader("🔍 Pairwise PCA Plot")
                fig = sns.pairplot(pca_df)
                for ax in fig.axes.flatten():
                    ax.set_facecolor('none')
                    ax.tick_params(colors='black')
                    ax.title.set_color('black')
                    ax.xaxis.label.set_color('black')
                    ax.yaxis.label.set_color('black')
                fig.fig.set_size_inches(5, 3)  # Reduce pairplot size
                fig.fig.patch.set_alpha(0.0)
                st.pyplot(fig)
                st.markdown(
                    "- Pairwise scatterplots illustrate relationships between chosen PCA components.\n"
                    "- Patterns or groupings detected here can inform further clustering or analysis."
                )

    st.subheader("🌐 Simulated 5G Network Load Example")
    x = np.arange(0, 24, 1)
    load = np.random.normal(loc=50, scale=10, size=len(x))
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
    plt.plot(x, load, marker='o', color='green')
    plt.title("Hourly 5G Traffic Load", color='black')
    plt.xlabel("Hour of Day", color='black')
    plt.ylabel("Network Load (%)", color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    st.pyplot(fig)
    st.markdown(
        "- This graph simulates how network load fluctuates across 24 hours in a 5G system.\n"
        "- Peaks and valleys represent varying user demand, useful for capacity planning."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("5G Hybrid Model Project")
