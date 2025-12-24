



import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load data
df = pd.read_csv('/Users/mac/Desktop/codes/unsupervised/dataset/mental-heath-in-tech-2016_20161114.csv')

print("Original shape:", df.shape)
with open(f"{output_dir}/01_original_info.txt", "w") as f:
    f.write(f"Original shape: {df.shape}\n")
    f.write("Original columns:\n")
    f.write("\n".join(df.columns.tolist()))

# Drop columns with too many missing values EARLY
df = df.loc[:, df.isnull().mean() < 0.85]

print("\nAfter dropping high-missing columns:", df.shape)
with open(f"{output_dir}/02_after_dropping_columns.txt", "w") as f:
    f.write(f"Shape after dropping columns with >=85% missing: {df.shape}\n")
    f.write("Remaining columns:\n")
    f.write("\n".join(df.columns.tolist()))

# Rename key columns (only if they exist)
rename_map = {
    'What is your age?': 'Age',
    'Do you have a family history of mental illness?': 'family_history',
    'Have you sought treatment for a mental health condition?': 'treatment',
    'If you have a mental health condition, do you feel that it interferes with your work?': 'work_interfere',
    'What is your gender?': 'Gender',
    'What country do you live in?': 'Country',
    'Are you self-employed?': 'self_employed',
    'Do you work remotely (outside of an office) at least 50% of the time?': 'remote_work',
}

existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
df = df.rename(columns=existing_rename)

print("\nAfter renaming:", df.columns.tolist())
with open(f"{output_dir}/03_columns_after_renaming.txt", "w") as f:
    f.write("Columns after renaming:\n")
    f.write("\n".join(df.columns.tolist()))

# Save descriptive statistics
desc = df.describe()
print(desc)
desc.to_csv(f"{output_dir}/04_descriptive_statistics.csv")

# EDA plot: Age distribution
if 'Age' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/05_age_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
else:
    print("Age column not available after dropping.")
    with open(f"{output_dir}/05_age_distribution.txt", "w") as f:
        f.write("Age column was dropped due to high missing values.")

# Step 2: Pre-processing

# Impute numerical
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_cols) > 0:
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute categorical with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Ordinal encoding for work_interfere if present
ordinal_cols = []
if 'work_interfere' in df.columns:
    order = [['Never', 'Rarely', 'Sometimes', 'Often']]
    oe = OrdinalEncoder(categories=order)
    df['work_interfere'] = oe.fit_transform(df[['work_interfere']])
    ordinal_cols = ['work_interfere']
else:
    print("'work_interfere' not available for ordinal encoding.")
    with open(f"{output_dir}/note.txt", "a") as f:
        f.write("'work_interfere' column was dropped (too many missing values).\n")

# One-hot encode remaining categorical
onehot_cols = [col for col in categorical_cols if col not in ordinal_cols]

if len(onehot_cols) > 0:
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = pd.DataFrame(
        ohe.fit_transform(df[onehot_cols]),
        columns=ohe.get_feature_names_out(onehot_cols),
        index=df.index
    )
    df = pd.concat([df.drop(onehot_cols, axis=1), encoded], axis=1)

# Scale
scaler = StandardScaler()
numerical_for_scaling = df.select_dtypes(include=['number']).columns
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_for_scaling]), columns=numerical_for_scaling)

# Step 3: Feature Engineering - Risk Score
risk_components = []
if 'family_history' in df.columns:
    risk_components.append(df['family_history'].map({'Yes': 1, 'No': 0, "Don't know": 0}))
if 'treatment' in df.columns:
    risk_components.append(df['treatment'].map({'Yes': 1, 'No': 0}))

if risk_components:
    df['risk_score'] = sum(risk_components)
else:
    print("Not enough columns for risk_score - skipping.")
    df['risk_score'] = 0

# Re-scale including risk_score
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include=['number'])),
    columns=df.select_dtypes(include=['number']).columns
)

# Step 4: PCA
pca = PCA(n_components=0.8)
df_pca = pca.fit_transform(df_scaled)

print(f"\nPCA: Reduced to {df_pca.shape[1]} components (explained variance: {pca.explained_variance_ratio_.sum():.3f})")

# Save PCA explained variance
pca_summary = pd.DataFrame({
    'Component': range(1, len(pca.explained_variance_ratio_) + 1),
    'Explained Variance Ratio': pca.explained_variance_ratio_,
    'Cumulative Variance': pca.explained_variance_ratio_.cumsum()
})
pca_summary.to_csv(f"{output_dir}/06_pca_explained_variance.csv", index=False)

# Step 5: Clustering
sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
labels = sc.fit_predict(df_pca)
silhouette = silhouette_score(df_pca, labels)
print(f"Silhouette Score: {silhouette:.3f}")

# Save clustering plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
plt.title('Spectral Clustering in PCA Space (3 Clusters)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.colorbar(scatter, label='Cluster')
plt.savefig(f"{output_dir}/07_clusters_in_pca_space.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Step 6: Interpretation
df['cluster'] = labels

interpretable_cols = ['Age', 'risk_score']
if 'work_interfere' in df.columns:
    interpretable_cols.append('work_interfere')
if 'remote_work' in df.columns:
    interpretable_cols.append('remote_work')
if 'family_history' in df.columns:
    interpretable_cols.append('family_history')
if 'treatment' in df.columns:
    interpretable_cols.append('treatment')

cluster_means = df.groupby('cluster')[interpretable_cols].mean()
print("\nCluster Characteristics (mean values per cluster):")
print(cluster_means)

# Save final results
cluster_means.to_csv(f"{output_dir}/08_cluster_characteristics.csv")
df.to_csv(f"{output_dir}/09_final_data_with_clusters.csv", index=False)

# Save summary report
with open(f"{output_dir}/00_summary_report.txt", "w") as f:
    f.write("Mental Health in Tech 2016 - Unsupervised Analysis Report\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset shape after cleaning: {df.shape}\n")
    f.write(f"Number of clusters: 3\n")
    f.write(f"Silhouette Score: {silhouette:.4f}\n")
    f.write(f"PCA components retained: {df_pca.shape[1]} (80% variance)\n")
    f.write(f"Interpretable features used: {', '.join(interpretable_cols)}\n\n")
    f.write("Cluster Profiles (means):\n")
    f.write(cluster_means.round(3).to_string())

print(f"\nAnalysis complete! All outputs saved to: {os.path.abspath(output_dir)}")