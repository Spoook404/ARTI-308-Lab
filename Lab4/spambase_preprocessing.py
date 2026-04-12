# Run In Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

columns = [
    'word_freq_make','word_freq_address','word_freq_all','word_freq_3d',
    'word_freq_our','word_freq_over','word_freq_remove','word_freq_internet',
    'word_freq_order','word_freq_mail','word_freq_receive','word_freq_will',
    'word_freq_people','word_freq_report','word_freq_addresses','word_freq_free',
    'word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
    'word_freq_your','word_freq_font','word_freq_000','word_freq_money',
    'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650',
    'word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
    'word_freq_data','word_freq_415','word_freq_85','word_freq_technology',
    'word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct',
    'word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project',
    'word_freq_re','word_freq_edu','word_freq_table','word_freq_conference',
    'char_freq_;','char_freq_(','char_freq_[','char_freq_!',
    'char_freq_$','char_freq_#',
    'capital_run_length_average','capital_run_length_longest',
    'capital_run_length_total',
    'class'
]

df = pd.read_csv("Lab3/spambase.csv", header=None)
df.columns = columns

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

#Data Quality Assestment
print("\n" + "=" * 60)
print("DATA QUALITY ASSESSMENT")
print("=" * 60)


print("\n--- 2.1 Data Types ---")
print(df.dtypes)
print("\nObservation: All features appear to be numeric (float64/int64), which is appropriate for this dataset.")


print("\n--- 2.2 Missing Values ---")
missing_values = df.isna().sum()
total_missing = df.isna().sum().sum()
print(f"Total missing values: {total_missing}")
print(f"\nMissing values per column:")
print(missing_values[missing_values > 0] if total_missing > 0 else "No missing values found!")


print("\n--- 2.3 Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print(f"Percentage of duplicates: {(duplicates/len(df))*100:.2f}%")


print("\n--- 2.4 Basic Statistics ---")
print(df.describe().T)

#Missing Values Handling
print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)


print("\nNote: Spambase dataset has no missing values.")
print("Demonstrating missing value handling strategies...")


df_missing_demo = df.copy()

df_missing_demo.loc[0:9, 'word_freq_free'] = np.nan

print(f"\nArtificial missing values introduced in 'word_freq_free':")
print(f"Missing values count: {df_missing_demo['word_freq_free'].isna().sum()}")

print("\n--- Strategy 1: Remove Records ---")
df_removed = df_missing_demo.dropna()
print(f"Original shape: {df_missing_demo.shape}")
print(f"After removing missing values: {df_removed.shape}")
print(f"Rows removed: {df_missing_demo.shape[0] - df_removed.shape[0]}")

print("\n--- Strategy 2: Mean Imputation ---")
df_mean_imputed = df_missing_demo.copy()
mean_value = df_mean_imputed['word_freq_free'].mean()
df_mean_imputed['word_freq_free'] = df_mean_imputed['word_freq_free'].fillna(mean_value)
print(f"Mean value used: {mean_value:.4f}")
print(f"Missing values after imputation: {df_mean_imputed['word_freq_free'].isna().sum()}")

print("\n--- Strategy 3: Median Imputation ---")
df_median_imputed = df_missing_demo.copy()
median_value = df_median_imputed['word_freq_free'].median()
df_median_imputed['word_freq_free'] = df_median_imputed['word_freq_free'].fillna(median_value)
print(f"Median value used: {median_value:.4f}")
print(f"Missing values after imputation: {df_median_imputed['word_freq_free'].isna().sum()}")

print("\n*** Recommendation for Spambase: ***")
print("Since the dataset has no missing values, no imputation is needed.")
print("If missing values existed, Mean Imputation would be suitable for normally distributed data,")
print("while Median Imputation would be better for skewed distributions.")


print("\n" + "=" * 60)
print("HANDLING OUTLIERS (IQR Method)")
print("=" * 60)

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('class')

print(f"Number of numerical features: {len(numerical_features)}")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_to_plot = ['word_freq_free', 'word_freq_you', 'word_freq_your', 
                    'char_freq_!', 'capital_run_length_longest', 'capital_run_length_total']

for i, feature in enumerate(features_to_plot):
    ax = axes[i//3, i%3]
    sns.boxplot(x=df[feature], ax=ax)
    ax.set_title(f"Boxplot of {feature}")

plt.tight_layout()
plt.savefig("Lab4/outliers_boxplot.png", dpi=100)
plt.show()


print("\n--- Outlier Detection for 'word_freq_free' ---")
Q1 = df['word_freq_free'].quantile(0.25)
Q3 = df['word_freq_free'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")

outliers = df[(df['word_freq_free'] < lower_bound) | (df['word_freq_free'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")

print("\n--- Outlier Summary for All Features ---")
outlier_summary = {}
for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_count = len(df[(df[feature] < lower) | (df[feature] > upper)])
    outlier_summary[feature] = outliers_count

outlier_df = pd.DataFrame(list(outlier_summary.items()), columns=['Feature', 'Outlier Count'])
outlier_df = outlier_df.sort_values('Outlier Count', ascending=False).head(10)
print(outlier_df.to_string(index=False))

print("\n--- Method 1: Remove Outliers ---")
df_no_outliers = df.copy()
for feature in numerical_features:
    Q1 = df_no_outliers[feature].quantile(0.25)
    Q3 = df_no_outliers[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_no_outliers = df_no_outliers[(df_no_outliers[feature] >= lower) & (df_no_outliers[feature] <= upper)]

print(f"Original shape: {df.shape}")
print(f"After removing outliers: {df_no_outliers.shape}")
print(f"Rows removed: {df.shape[0] - df_no_outliers.shape[0]}")

print("\n--- Method 2: Cap Outliers (Percentile Method) ---")
df_capped = df.copy()
for feature in numerical_features:
    lower_cap = df_capped[feature].quantile(0.05)
    upper_cap = df_capped[feature].quantile(0.95)
    df_capped[feature] = df_capped[feature].clip(lower_cap, upper_cap)

print(f"Dataset shape after capping: {df_capped.shape}")
print("\n*** Note: For spam detection, extreme values may represent important patterns.")
print("We recommend using the capped dataset to preserve rare events.")

#Normalization
print("\n" + "=" * 60)
print("DATA TRANSFORMATION - NORMALIZATION")
print("=" * 60)

df_normalized = df_capped.copy()

features_to_normalize = [col for col in numerical_features]

print("\n--- 5.1 Min-Max Normalization ---")
print("Scales values to range [0, 1]")
print("Formula: X_scaled = (X - X_min) / (X_max - X_min)")

min_max_scaler = MinMaxScaler()
df_minmax = df_normalized[features_to_normalize].copy()
df_minmax[features_to_normalize] = min_max_scaler.fit_transform(df_normalized[features_to_normalize])

print(f"\nOriginal data range (example - word_freq_free): {df_normalized['word_freq_free'].min():.2f} to {df_normalized['word_freq_free'].max():.2f}")
print(f"Normalized data range (word_freq_free): {df_minmax['word_freq_free'].min():.4f} to {df_minmax['word_freq_free'].max():.4f}")

print("\nMin-Max Normalization Sample (first 5 rows):")
print(df_minmax[['word_freq_free', 'word_freq_you', 'capital_run_length_average']].head())

print("\n--- 5.2 Z-Score Standardization ---")
print("Transforms data to have mean=0 and std=1")
print("Formula: X_standardized = (X - mean) / std")

standard_scaler = StandardScaler()
df_standardized = df_normalized[features_to_normalize].copy()
df_standardized[features_to_normalize] = standard_scaler.fit_transform(df_normalized[features_to_normalize])

print(f"\nOriginal mean (word_freq_free): {df_normalized['word_freq_free'].mean():.2f}")
print(f"Standardized mean (word_freq_free): {df_standardized['word_freq_free'].mean():.6f}")
print(f"Standardized std (word_freq_free): {df_standardized['word_freq_free'].std():.6f}")

print("\nZ-Score Standardization Sample (first 5 rows):")
print(df_standardized[['word_freq_free', 'word_freq_you', 'capital_run_length_average']].head())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df_normalized['word_freq_free'], bins=30, edgecolor='black')
axes[0, 0].set_title("Original Distribution - word_freq_free")
axes[0, 0].set_xlabel("Value")

axes[0, 1].hist(df_minmax['word_freq_free'], bins=30, edgecolor='black', color='green')
axes[0, 1].set_title("Min-Max Normalized - word_freq_free")
axes[0, 1].set_xlabel("Value (0-1)")

axes[1, 0].hist(df_standardized['word_freq_free'], bins=30, edgecolor='black', color='orange')
axes[1, 0].set_title("Z-Score Standardized - word_freq_free")
axes[1, 0].set_xlabel("Z-Score")

axes[1, 1].hist(df_normalized['capital_run_length_longest'], bins=30, alpha=0.5, label='Original', edgecolor='black')
axes[1, 1].hist(df_minmax['capital_run_length_longest'], bins=30, alpha=0.5, label='Min-Max', edgecolor='black')
axes[1, 1].set_title("Comparison - capital_run_length_longest")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("Lab4/normalization_comparison.png", dpi=100)
plt.show()

print("\n*** When to use which normalization: ***")
print("- Min-Max: For distance-based algorithms (KNN, K-Means, SVM)")
print("- Z-Score: For algorithms assuming normal distribution (Linear Regression, Logistic Regression, PCA)")

#Correlation Analysis
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

corr_matrix = df_standardized.corr()

print("\n--- Highly Correlated Features (|r| > 0.7) ---")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df.to_string(index=False))
else:
    print("No feature pairs with correlation > 0.7 found.")

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
            annot=False, square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap (Standardized Data)", fontsize=14)
plt.tight_layout()
plt.savefig("Lab4/correlation_heatmap.png", dpi=100)
plt.show()

print("\n*** Correlation Analysis Insight: ***")
print("Features with high correlation contain overlapping information.")
print("PCA can help reduce dimensionality by combining correlated features.")

#PCA
print("\n" + "=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 60)

X = df_standardized[features_to_normalize]
y = df['class']  


pca_full = PCA()
pca_full.fit(X)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("--- Explained Variance by Component ---")
print(f"Number of components: {len(explained_variance_ratio)}")

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
print(f"Components needed for 99% variance: {n_components_99}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7, label='Individual')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Explained Variance by Component')
axes[0].set_xlim(0, 60)

axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
axes[1].axvline(x=n_components_95, color='r', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].set_xlim(0, 60)

plt.tight_layout()
plt.savefig("Lab4/pca_variance.png", dpi=100)
plt.show()

print(f"\n--- Applying PCA with {n_components_95} components (95% variance) ---")
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"\nExplained variance ratio by component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"\nTotal variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
plt.colorbar(scatter, label='Class (0=Non-Spam, 1=Spam)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.title('PCA Projection of Spambase Dataset')
plt.savefig("Lab4/pca_projection.png", dpi=100)
plt.show()

#Summary
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print("""
Dataset: Spambase (from Lab 3)

Data Quality Assessment:
- All features are numeric (appropriate for the dataset)
- No missing values detected
- Found {} duplicate rows

Missing Values Handling:
- No missing values to handle
- Demonstrated Mean, Median, and Removal strategies

Outlier Handling:
- Used IQR method to detect outliers
- Applied capping (5th-95th percentile) to preserve rare events
- Removed {} rows when using complete outlier removal

Normalization:
- Min-Max: Scales to [0,1] range
- Z-Score: Centers at 0 with std=1
- Both preserve relative relationships between data points

PCA:
- Reduced from {} features to {} components (95% variance)
- Retains {}% of total variance
- Useful for visualization and reducing computational cost
""".format(
    duplicates,
    df.shape[0] - df_no_outliers.shape[0],
    len(features_to_normalize),
    n_components_95,
    sum(pca.explained_variance_ratio_)*100
))

print("=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)

print("\nSaving preprocessed datasets...")
df_standardized.to_csv("Lab4/spambase_standardized.csv", index=False)
df_minmax.to_csv("Lab4/spambase_minmax.csv", index=False)
df_capped.to_csv("Lab4/spambase_capped.csv", index=False)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components_95)])
pca_df['class'] = y.values
pca_df.to_csv("Lab4/spambase_pca.csv", index=False)

print("Files saved:")
print("  - Lab4/spambase_standardized.csv (Z-Score standardized)")
print("  - Lab4/spambase_minmax.csv (Min-Max normalized)")
print("  - Lab4/spambase_capped.csv (Outlier capped)")
print("  - Lab4/spambase_pca.csv (PCA transformed)")
