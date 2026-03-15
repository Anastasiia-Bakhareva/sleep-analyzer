import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. LOAD DATA
# =====================
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
print("Data loaded:", df.shape)

# =====================
# 2. DATA PREPARATION
# =====================

# Split Blood Pressure into Systolic and Diastolic
df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)

# Create Sleep Efficiency (sleep duration as % of 24 hours)
df["Sleep_Efficiency"] = (df["Sleep Duration"] / 24 * 100).round(2)

# Fix BMI Category inconsistency (Normal Weight -> Normal)
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")

# Fill missing Sleep Disorder values with "None"
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# Create binary target variable: 1 = good sleep (>=6), 0 = poor sleep (<6)
df["Quality_Sleep_Binary"] = (df["Quality of Sleep"] >= 6).astype(int)

print("\nAfter preparation:")
print(df[["Sleep Duration", "Systolic_BP", "Diastolic_BP", "Sleep_Efficiency", "Quality_Sleep_Binary"]].head())

# =====================
# 3. SAVE TO SQL
# =====================
conn = sqlite3.connect("sleep.db")
df.to_sql("sleep_data", conn, if_exists="replace", index=False)
print("\nData saved to database!")

# SQL queries
print("\n--- SQL QUERIES ---")

avg_sleep = pd.read_sql("SELECT ROUND(AVG([Sleep Duration]), 2) as avg_sleep FROM sleep_data", conn)
print("\nAverage sleep duration:")
print(avg_sleep)

stress_by_bmi = pd.read_sql("""
    SELECT [BMI Category], ROUND(AVG([Stress Level]), 2) as avg_stress
    FROM sleep_data
    GROUP BY [BMI Category]
    ORDER BY avg_stress DESC
""", conn)
print("\nAverage stress level by BMI category:")
print(stress_by_bmi)

disorder_count = pd.read_sql("""
    SELECT [Sleep Disorder], COUNT(*) as count
    FROM sleep_data
    GROUP BY [Sleep Disorder]
    ORDER BY count DESC
""", conn)
print("\nSleep disorder distribution:")
print(disorder_count)

conn.close()

# =====================
# 4. VISUALIZATIONS
# =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sleep Health and Lifestyle Analysis", fontsize=16)

# Chart 1: Sleep Duration distribution
axes[0, 0].hist(df["Sleep Duration"], bins=15, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Sleep Duration Distribution")
axes[0, 0].set_xlabel("Hours")
axes[0, 0].set_ylabel("Count")

# Chart 2: Stress Level by BMI Category
sns.boxplot(data=df, x="BMI Category", y="Stress Level", hue="BMI Category", ax=axes[0, 1], palette="Set2", legend=False)
axes[0, 1].set_title("Stress Level by BMI Category")

# Chart 3: Sleep Disorder distribution
disorder_counts = df["Sleep Disorder"].value_counts()
axes[1, 0].bar(disorder_counts.index, disorder_counts.values, color=["steelblue", "orange", "red"])
axes[1, 0].set_title("Sleep Disorder Distribution")
axes[1, 0].set_ylabel("Count")

# Chart 4: Sleep Duration vs Quality of Sleep
axes[1, 1].scatter(df["Sleep Duration"], df["Quality of Sleep"], alpha=0.5, color="steelblue")
axes[1, 1].set_title("Sleep Duration vs Quality of Sleep")
axes[1, 1].set_xlabel("Sleep Duration (hours)")
axes[1, 1].set_ylabel("Quality of Sleep")

plt.tight_layout()
plt.savefig("sleep_analysis.png")
plt.show()
print("\nChart saved as sleep_analysis.png")

# =====================
# 5. LOGISTIC REGRESSION
# =====================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Select features
features = ["Sleep Duration", "Physical Activity Level", "Stress Level", "Age"]
X = df[features]
y = df["Quality_Sleep_Binary"]

# Split into train and test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- LOGISTIC REGRESSION ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

