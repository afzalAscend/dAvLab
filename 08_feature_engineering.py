import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

data = {
    "Study_Hours": [2, 4, 6, 8, 10, 3, 5, 7, 9, 12],
    "Attendance": [60, 65, 70, 75, 80, 62, 68, 78, 85, 95],
    "Assignment_Score": [40, 50, 55, 65, 70, 45, 60, 72, 80, 90],
    "Exam_Score": [45, 50, 60, 65, 75, 48, 58, 70, 82, 95]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\nStatistical Summary:")
print(df.describe())

minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print("\nNormalized Dataset:")
print(normalized_df.round(2))
print("\nMinimum Values:")
print(normalized_df.min())
print("\nMaximum Values:")
print(normalized_df.max())

standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

print("\nStandardized Dataset:")
print(standardized_df.round(2))
print("\nMean of Each Feature:")
print(standardized_df.mean().round(2))
print("\nStandard Deviation of Each Feature:")
print(standardized_df.std(ddof=0).round(2))

pca_scaler = StandardScaler()
scaled_data = pca_scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

print("\nPCA Transformed Data:")
print(pca_df.round(2))
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nTotal Variance Retained:")
print(pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8, 5))
plt.scatter(pca_df["PC1"], pca_df["PC2"], color="blue", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Dimensionality Reduction")
plt.grid(True)
plt.show()