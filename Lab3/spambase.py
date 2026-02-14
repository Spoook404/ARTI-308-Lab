import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

df = pd.read_csv("spambase.csv", header=None)
df.columns = columns



print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget value counts:")
print(df["class"].value_counts())


print("Missing values per column:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df.head()

df.describe().T
print(df.isna().sum())

df['word_freq_free'].hist(bins=30)
plt.title("Distribution of 'free' word frequency")
plt.show()

df['capital_run_length_longest'].hist(bins=30)
plt.title("Distribution of longest capital run")
plt.show()

df.groupby('class')['word_freq_free'].mean()
df.groupby('class')['char_freq_!'].mean()
df.groupby('class')['capital_run_length_longest'].mean()

df.boxplot(column='word_freq_free', by='class')
plt.title("Word 'free' Frequency by Class")
plt.suptitle("")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()