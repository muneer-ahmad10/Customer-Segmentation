# Customer-Segmentation
Customer segmentation revealed three key groups: low-value customers, high-value older customers, and young high-spending customers, indicating different behavioral patterns across age groups

# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project focuses on segmenting customers based on their purchasing behavior using **unsupervised machine learning (K-Means Clustering)**.

The main objective was not just to apply a clustering algorithm, but to understand the **importance of data preprocessing, feature engineering, and meaningful interpretation** in real-world data science problems.

---

## 🎯 Objective
To identify meaningful customer segments based on behavior and provide **actionable business insights**.

---

## 📂 Dataset Description
The dataset was created by merging three different sources:

- **Customer Data**
  - `customer_id`, `gender`, `age`, `payment_method`

- **Sales Data**
  - `invoice_no`, `customer_id`, `category`, `quantity`, `price`, `shopping_mall`

- **Shopping Mall Data**
  - `shopping_mall`, `construction_year`, `area`, `location`, `store_count`

---

## 🔗 Data Merging
Datasets were merged using:
- `customer_id` → to combine customer and sales data  
- `shopping_mall` → to add mall-level information  

👉 After merging, the dataset contained **transaction-level data** (multiple rows per customer).

---

## 🔄 Data Transformation (Important Step)

To perform clustering, the dataset was transformed into:

👉 **Customer-level data (1 row = 1 customer)**

### Created Features:
- `total_spent` → total money spent by customer  
- `quantity` → total items purchased  
- `purchase_count` → number of transactions  
- `store_count` → number of stores in visited mall  

---

## 🧹 Data Cleaning
- Dropped unnecessary columns:
  - `customer_id`, `invoice_no`, text-based columns  
- Handled missing values:
  - `age` → filled with median  
  - `store_count` → filled with median  

---

## 🎯 Feature Selection

Not all features contribute to clustering.

### ❌ Dropped:
- `quantity` (low variance)  
- `purchase_count` (almost constant)  
- identifiers and categorical noise  

### ✅ Selected Features:
- `age`  
- `total_spent`  
- `store_count`  

---

## ⚠️ Handling Skewness

The `total_spent` feature was highly right-skewed.

### Applied transformation:
```python
df['total_spent'] = np.log1p(df['total_spent'])

📊 Exploratory Data Analysis (EDA)
Key Observations:
Majority of customers are low spenders
Few customers have very high spending (outliers)
No clear visible clusters → required algorithmic clustering
📏 Feature Scaling

Since K-Means is distance-based, scaling is necessary:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
📊 Finding Optimal Clusters
🔍 Elbow Method

Used to identify optimal number of clusters

📈 Silhouette Score Results:
K	Score
2	0.245
3	0.250
4	0.273
5	0.262
6	0.281
⚖️ Final Model Selection
K = 6 → Best mathematically
K = 3 → Best interpretability

👉 Final Choice: K = 3

Reason:

Easier to interpret
More meaningful business insights
🤖 Model Implementation
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)
📊 Customer Segments (Results)
🔴 Cluster 0 → Low Value Customers
Moderate age
Low spending

💡 Strategy:

Discounts
Engagement campaigns
🟢 Cluster 1 → High Value Older Customers
Older age group
High spending

💡 Strategy:

Loyalty programs
Premium services
🔵 Cluster 2 → Young High-Spending Customers
Younger customers
High spending

💡 Strategy:

Targeted ads
Personalized recommendations
💡 Key Insights
Customers are not homogeneous → clear segmentation exists
High-value customers are split across different age groups
Feature engineering plays a bigger role than algorithm complexity
More clusters do not always lead to better business understanding
🛠️ Tech Stack
Python
Pandas
NumPy
Matplotlib / Seaborn
Scikit-learn
🚀 Future Improvements
Include time-based features (Recency, Frequency, Monetary)
Try advanced clustering (DBSCAN, Hierarchical)
Use PCA for dimensionality reduction
Deploy as an interactive dashboard
📌 Conclusion

This project demonstrates that:

👉 The quality of insights in machine learning depends more on data preparation and feature engineering than on the algorithm itself.
