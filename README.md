# Customer Segmentation Using K-Means Clustering

### Project Overview

This project aims to segment customers based on their purchasing behavior and demographic information. The segmentation will help the retail company develop targeted marketing strategies, personalize product recommendations, and optimize pricing strategies.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the required packages.

```bash
git clone https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering.git
cd Customer-Segmentation-Using-K-Means-Clustering
pip install -r requirements.txt
```

## Usage

To run the analysis and generate the clusters, execute the following script:

```bash
python main.py
```

## Data

The dataset used for this project is provided in the data directory. It contains information about customer demographics and spending habits:

- CustomerID: Unique ID for each customer
- Gender: Gender of the customer
- Age: Age of the customer
- Annual Income: Annual income of the customer
- Spending Score: Spending score assigned by the retail company

## Data Cleaning and Pre-Processing

The data is cleaned and has no missing or duplicate values. However, the redundant column, 'CustomerID' was dropped and the misspelled column, 'Genre' was renamed.

```python
# Drop CustomerID as it is not needed for clustering

data = data.drop('CustomerID', axis=1)
# Renaming the 'Genre' column to 'Gender'

data=data.rename(columns={"Genre": "Gender"})
```

```python
# Categorizing Age bracket
def Age_group(Age):
    if Age <= 19:
        return "Teenager"
    elif Age <= 35:
        return "Youth"
    elif Age <= 55:
        return "Middle-Age"
    else:
        return "Elder"
data['Age_group'] = data['Age'].apply(Age_group)
```

## Exploratory Data Analysis (EDA)

Before applying the clustering algorithm, an EDA was performed to understand the distribution of the data and identify any patterns.

Example of a box plot for Age_group vs Spending_Score:

```python
# Create a box plot for Age_group vs Spending_Score
plt.figure(figsize=(12, 6))
sns.boxplot(x='Age_group', y='Spending_Score', data=data)
plt.title('Box Plot of Age Group vs Spending Score')
plt.xlabel('Age Group')
plt.ylabel('Spending Score')
plt.show()
```

![image](https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering/assets/82883963/7dc8b381-49d2-4341-8289-ad3f622d23fc)


The boxplot revealed that younger customers tend to have higher spending scores compared to older customers.

- Examine relationships between different features

We created scatter plots and pair plots to examine relationships between different features:

```python
# Scatter plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x='Age', y='Annual_Income_(k$)', hue='Gender', data=data)
plt.title('Age vs Annual Income')

plt.subplot(1, 3, 2)
sns.scatterplot(x='Age', y='Spending_Score', hue='Gender', data=data)
plt.title('Age vs Spending Score')

plt.subplot(1, 3, 3)
sns.scatterplot(x='Annual_Income_(k$)', y='Spending_Score', hue='Gender', data=data)
plt.title('Annual_Income_(k$) vs Spending Score')

plt.tight_layout()
plt.show()

# Pair plot
sns.pairplot(data, hue='Gender')
plt.show()
```
![image](https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering/assets/82883963/b3e42e3e-ed39-4c8d-a4b2-3b4364df5ace)

![image](https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering/assets/82883963/08d90af1-96ad-4c50-a40f-018f634feb2b)

- The scatter plots reveal correlations, such as higher income customers having varied spending scores, indicating different spending habits.
- Customers with higher annual income do not necessarily have higher spending scores.
- Certain age groups have higher spending scores, indicating more active spending behavior.

```python
from sklearn.preprocessing import LabelEncoder

# Encode Gender as a numerical variable
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Drop 'Age_group as it is not needed for clustering
data = data.drop('Age_group', axis=1)

data.head()
```

## Modeling

We used the K-Means clustering algorithm to segment the customers. The optimal number of clusters was determined using the elbow method and the silhouette score.

Example code for applying K-Means with 5 clusters:

```python
from sklearn.cluster import KMeans
# Determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss, color='red', marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

![image](https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering/assets/82883963/df80301b-8f4a-4446-b758-a00e344ad771)

The elbow method graph shows a noticeable "elbow" k=5, which suggests that 5 clusters might be optimal.

## Evaluation

The silhouette score was used to evaluate the clustering performance.

Example code to calculate the silhouette score:

```python
# Calculate silhouette score
sil_score = silhouette_score(data[['Annual_Income_(k$)', 'Spending_Score']], data['Cluster'])
print(f'Silhouette Score for 5 Clusters: {score}')
```

- **Silhouette Score: 0.5503719213912603:**

The silhouette score ranges from -1 to 1, where a higher score indicates better-defined clusters. A score above 0.5 suggests that the clusters are reasonably well-separated and well-defined.
With a silhouette score of approximately 0.55, it can be concluded that the clustering model has performed quite well in identifying distinct customer segments. However, there's still room for improvement, as an ideal score would be closer to 1.

## Results

The clustering results were visualized using scatter plots. Below is an example of the clusters based on Annual_Income and Spending_Score:

```python
# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Income_(k$)', y='Spending_Score', hue='Cluster', palette='viridis', data=data, s=100)
plt.title('Customer Segments (5 Clusters)')
plt.xlabel('Annual_Income_(k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')

plt.show()
```
![image](https://github.com/GogoHarry/Customer-Segmentation-Using-K-Means-Clustering/assets/82883963/17ed89a1-1ca9-4dd7-913e-da2852d69fc1)


1. **Cluster Characteristics**:
   - **Cluster 0 (Purple)**: This cluster includes customers with a wide range of annual incomes (from around 50k to over 140k$) but generally lower spending scores (0-40).
   
   - **Cluster 1 (Blue)**: This cluster contains customers with lower to moderate annual incomes (about 10k to 40k) but high spending scores (60-100).
   - **Cluster 2 (Teal)**: These customers have a wide range of annual incomes (70k to about 140k$) and high spending scores (60-100).
   - **Cluster 3 (Green)**: This cluster includes customers with lower to moderate annual incomes (10k to about 40k$) and low to moderate spending scores (0-40).
   - **Cluster 4 (Yellow)**: Customers in this cluster have moderate annual incomes (40k to 70k$) and moderate spending scores (40-60).

2. **Income vs. Spending Behavior**:
   - High spenders (high spending scores) tend to fall into distinct clusters based on their annual income levels. For example, Cluster 1 and Cluster 2 both have high spending scores but different income ranges.

## Conclusion

The clustering analysis has successfully identified five distinct customer segments based on annual income and spending scores. The relatively high silhouette score indicates that the clusters are well-defined and meaningful, providing valuable insights for targeted marketing strategies and customer relationship management. Future efforts might focus on refining the clustering process or exploring additional features to further enhance cluster separation and understanding.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - See the [LICENSE](LICENSE) file for details.
