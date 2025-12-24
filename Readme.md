# Mental Health in Tech 2016 - Unsupervised Clustering Analysis

This project performs an exploratory unsupervised analysis on the **OSMI Mental Health in Tech Survey 2016** dataset using dimensionality reduction (PCA) and clustering (Spectral Clustering). The goal is to identify hidden patterns or subgroups among respondents based on demographic, work, and mental health-related features.

## Dataset

- **Source**: OSMI Mental Health in Tech Survey 2016  
- **File**: `mental-heath-in-tech-2016_20161114.csv`  
- **Location**: `./dataset/`  
- **Rows**: ~1,429 responses  
- **Original Columns**: 63 (mostly survey questions with long text headers)

The dataset contains many columns with high missing rates due to skip logic (e.g., questions about employer benefits are only answered by non-self-employed respondents).

## Project Objective

Apply unsupervised machine learning to:
- Clean and preprocess the highly sparse survey data
- Engineer meaningful features (e.g., a simple risk score)
- Reduce dimensionality with PCA
- Cluster respondents using Spectral Clustering
- Interpret the resulting clusters

## Requirements

Python 3.11+ and the following packages (see `requirements.txt`):

```txt
pandas>=2.3.0
scikit-learn>=1.8.0
matplotlib>=3.9.0
seaborn>=0.13.0
```
Install with
```bash
 pip install -r requirements.txt
```

## Project Structure

```bash
├── dataset/
│   └── mental-heath-in-tech-2016_20161114.csv    # Original dataset
├── src/
│   └── mental_health_analysis.py 
|    ├── requirements.txt
|   └── README.md         
├── analysis_output/                             # Created automatically
│   ├── 00_summary_report.txt
│   ├── 01_original_info.txt
│   ├── 02_after_dropping_columns.txt.            #analysis_output will display after running the code
│   ├── 03_columns_after_renaming.txt
│   ├── 04_descriptive_statistics.csv
│   ├── 05_age_distribution.png
│   ├── 06_pca_explained_variance.csv
│   ├── 07_clusters_in_pca_space.png
│   ├── 08_cluster_characteristics.csv
│   └── 09_final_data_with_clusters.csv

```

### How to Run

Follow these steps to execute the analysis:

1. **Place the dataset**  
   Copy the survey file into the project folder with this exact name and path:  
   `./dataset/mental-heath-in-tech-2016_20161114.csv`

2. **Activate your virtual environment** (recommended)  
   ```bash
   # Example for venv
   source venv/bin/activate    # On macOS/Linux
   # or
   venv\Scripts\activate       # On Windows

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ``` 
4. **Run the analysis script**
   ```bash
   python src/mental_health_analysis.py
   ```

### Pipeline Overview

#### 1. Data Loading & Initial Cleaning
- Load the survey CSV file
- Drop columns with ≥85% missing values (to handle skip-logic sparsity while retaining important columns like work interference)
- Rename key long survey question columns to short, meaningful names (e.g., `What is your age?` → `Age`)

#### 2. Exploratory Data Analysis
- Compute and display basic descriptive statistics
- Generate and save an **Age distribution plot** (histogram with KDE) as PNG

#### 3. Preprocessing
- **Numerical columns**: KNN imputation (n_neighbors=5)
- **Categorical columns**: Mode imputation
- **Ordinal encoding** for `work_interfere` using logical order:  
  `Never` → `Rarely` → `Sometimes` → `Often`
- **One-hot encoding** for all remaining categorical variables (with `drop='first'` to avoid multicollinearity)
- **Standard scaling** applied to all features

#### 4. Feature Engineering
- Create a simple additive `risk_score`:  
  `risk_score = (family_history == Yes ? 1 : 0) + (treatment == Yes ? 1 : 0)`

#### 5. Dimensionality Reduction
- Apply **PCA** retaining **80% of explained variance**
- Output number of components needed and cumulative variance

#### 6. Clustering
- **Spectral Clustering** with fixed `n_clusters=3` and `affinity='nearest_neighbors'`
- Compute and report **Silhouette Score** for cluster quality

#### 7. Interpretation & Output
- Generate **scatter plot** of the first two principal components colored by cluster labels
- Compute **mean values** of interpretable features for each cluster
- Save the full processed dataset with added `cluster` column

### Key Outputs (in `analysis_output/` folder)

- Summary Report (00_summary_report.txt): High-level overview
- Age Distribution (05_age_distribution.png): Histogram + KDE
- Cluster Visualization (07_clusters_in_pca_space.png): Main result
- Cluster Profiles (08_cluster_characteristics.csv): Mean Age, risk_score, work interference, remote work, etc., per cluster
- Full Data (09_final_data_with_clusters.csv): Original + processed features + cluster labels



### Notes & Limitations
- The dataset contains significant skip-logic missingness; aggressive column dropping (≥85% missing) is applied to ensure a clean feature set.
- Number of clusters is hard-coded to 3 (can be changed by modifying `n_clusters` in the script).
- Spectral Clustering is chosen for its strength in detecting non-convex clusters; alternatives like KMeans or DBSCAN may yield different insights.
- One-hot encoding of `Country` creates many columns due to global respondent distribution.

### Future Improvements
- Experiment with different clustering algorithms and determine optimal cluster count (e.g., elbow method, silhouette analysis across k values)
- Develop more sophisticated feature engineering (e.g., additional composite scores)
- Add interactive visualizations using Plotly
- Enhance cluster profiling with categorical breakdowns (e.g., percentage of remote workers, gender distribution per cluster)

### License
This project is for **educational and research purposes**.  
The original dataset is publicly available from the **Open Sourcing Mental Illness (OSMI)** Mental Health in Tech Survey.





