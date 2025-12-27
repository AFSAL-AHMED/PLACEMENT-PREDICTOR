import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("STUDENT PLACEMENT PREDICTION - ML PROJECT")
print("="*80)

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")
print("\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Drop serial number column
if 'sl_no' in df.columns:
    df = df.drop(columns=["sl_no"])
    print("\nDropped 'sl_no' column")

print(f"New shape: {df.shape}")

# Check missing values
print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if 'salary' in df.columns and df['salary'].isnull().sum() > 0:
    print(f"\nNote: {df['salary'].isnull().sum()} null values in salary column")
    print("(NULL salary = student not placed)")

# Check duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Removed duplicates. New shape: {df.shape}")

# Identify column types
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

print("\nSummary statistics:")
print(df.describe())

print("\n" + "="*80)
print("LABEL ENCODING")
print("="*80)

print("\nConverting text to numbers...")
print(f"Encoding {len(categorical_cols)} categorical columns")

le = LabelEncoder()
label_encoders = {}
df_before_encoding = df.copy()

for col in categorical_cols:
    print(f"\nEncoding {col}...")
    print(f"  Values: {df[col].unique()[:3]}")
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded to: {df[col].unique()[:3]}")

print("\nAll columns encoded successfully!")

# Show encoding mappings for key columns
print("\nEncoding Mappings:")
important_cols = ['gender', 'workex', 'status', 'specialisation']
for col in important_cols:
    if col in df_before_encoding.columns:
        temp_encoder = LabelEncoder()
        temp_encoder.fit(df_before_encoding[col])
        print(f"\n{col.upper()}:")
        for i, val in enumerate(temp_encoder.classes_):
            print(f"  {val} -> {i}")

print("\n" + "="*80)
print("TRAIN/TEST SPLIT")
print("="*80)

# Drop salary column
if 'salary' in df.columns:
    df_for_model = df.drop(columns=['salary'])
    print("\nDropped 'salary' column (to avoid data leakage)")
else:
    df_for_model = df.copy()

print(f"Features for model: {df_for_model.shape[1]} columns")

# Separate features and target
if 'status' in df_for_model.columns:
    y = df_for_model['status']
    X = df_for_model.drop(columns=['status'])
    print(f"\nFeatures (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"\nFeature list: {list(X.columns)}")

# Check distribution
print("\nPlacement distribution:")
placement_counts = y.value_counts().sort_index()
for class_val, count in placement_counts.items():
    class_name = "Not Placed" if class_val == 0 else "Placed"
    percentage = (count / len(y)) * 100
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

# Split data
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Show split distribution
train_distribution = y_train.value_counts().sort_index()
test_distribution = y_test.value_counts().sort_index()

print("\nTrain distribution:")
for class_val, count in train_distribution.items():
    class_name = "Not Placed" if class_val == 0 else "Placed"
    print(f"  {class_name}: {count}")

print("\nTest distribution:")
for class_val, count in test_distribution.items():
    class_name = "Not Placed" if class_val == 0 else "Placed"
    print(f"  {class_name}: {count}")

print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

print("\nTraining 2 models for comparison...")

# Logistic Regression
print("\n" + "-"*80)
print("Model 1: Logistic Regression")
print("-"*80)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
start_time = time.time()
lr_model.fit(X_train, y_train)
training_time = time.time() - start_time

y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)

print(f"\nTraining time: {training_time:.4f} seconds")
print(f"Train accuracy: {train_accuracy_lr*100:.2f}%")
print(f"Test accuracy: {test_accuracy_lr*100:.2f}%")

# Random Forest
print("\n" + "-"*80)
print("Model 2: Random Forest")
print("-"*80)

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
)

start_time = time.time()
rf_model.fit(X_train, y_train)
training_time_rf = time.time() - start_time

y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\nTraining time: {training_time_rf:.4f} seconds")
print(f"Train accuracy: {train_accuracy_rf*100:.2f}%")
print(f"Test accuracy: {test_accuracy_rf*100:.2f}%")

# Feature importance
print("\nFeature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

# Compare models
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Train Acc (%)', 'Test Acc (%)', 'Training Time (s)'],
    'Logistic Regression': [
        f'{train_accuracy_lr*100:.2f}',
        f'{test_accuracy_lr*100:.2f}',
        f'{training_time:.4f}'
    ],
    'Random Forest': [
        f'{train_accuracy_rf*100:.2f}',
        f'{test_accuracy_rf*100:.2f}',
        f'{training_time_rf:.4f}'
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Determine best model
if test_accuracy_rf > test_accuracy_lr:
    best_model = "Random Forest"
    best_accuracy = test_accuracy_rf
    best_model_obj = rf_model
else:
    best_model = "Logistic Regression"
    best_accuracy = test_accuracy_lr
    best_model_obj = lr_model

print(f"\nBest Model: {best_model}")
print(f"Best Accuracy: {best_accuracy*100:.2f}%")

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_test_pred_lr)
cm_rf = confusion_matrix(y_test, y_test_pred_rf)

print("\nLogistic Regression - Confusion Matrix:")
print(f"\n              Predicted")
print(f"            Not Placed  Placed")
print(f"Actual Not P     {cm_lr[0][0]}        {cm_lr[0][1]}")
print(f"       Placed    {cm_lr[1][0]}        {cm_lr[1][1]}")

tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
print(f"\nTN: {tn_lr}, FP: {fp_lr}, FN: {fn_lr}, TP: {tp_lr}")

print("\nRandom Forest - Confusion Matrix:")
print(f"\n              Predicted")
print(f"            Not Placed  Placed")
print(f"Actual Not P     {cm_rf[0][0]}        {cm_rf[0][1]}")
print(f"       Placed    {cm_rf[1][0]}        {cm_rf[1][1]}")

tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
print(f"\nTN: {tn_rf}, FP: {fp_rf}, FN: {fn_rf}, TP: {tp_rf}")

# Classification Report
print("\n" + "-"*80)
print("Classification Report - Logistic Regression")
print("-"*80)
print(classification_report(y_test, y_test_pred_lr, target_names=['Not Placed', 'Placed']))

print("\n" + "-"*80)
print("Classification Report - Random Forest")
print("-"*80)
print(classification_report(y_test, y_test_pred_rf, target_names=['Not Placed', 'Placed']))

# Calculate metrics
precision_lr = precision_score(y_test, y_test_pred_lr)
recall_lr = recall_score(y_test, y_test_pred_lr)
f1_lr = f1_score(y_test, y_test_pred_lr)

precision_rf = precision_score(y_test, y_test_pred_rf)
recall_rf = recall_score(y_test, y_test_pred_rf)
f1_rf = f1_score(y_test, y_test_pred_rf)

# Create visualizations
print("\nCreating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'],
            ax=axes[0], cbar=True)
axes[0].set_title(f'Logistic Regression\nAccuracy: {test_accuracy_lr*100:.2f}%', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'],
            ax=axes[1], cbar=True)
axes[1].set_title(f'Random Forest\nAccuracy: {test_accuracy_rf*100:.2f}%', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")
plt.close()

# Performance comparison chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_scores = [test_accuracy_lr*100, precision_lr*100, recall_lr*100, f1_lr*100]
rf_scores = [test_accuracy_rf*100, precision_rf*100, recall_rf*100, f1_rf*100]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='skyblue')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightgreen')

ax.set_xlabel('Metrics', fontweight='bold')
ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save best model
model_filename = 'placement_prediction_model.pkl'
print(f"\nSaving {best_model} model...")
with open(model_filename, 'wb') as file:
    pickle.dump(best_model_obj, file)
print(f"Saved: {model_filename}")

# Save encoders
encoders_filename = 'label_encoders.pkl'
with open(encoders_filename, 'wb') as file:
    pickle.dump(label_encoders, file)
print(f"Saved: {encoders_filename}")

# Save feature names
feature_names_file = 'feature_names.pkl'
with open(feature_names_file, 'wb') as file:
    pickle.dump(list(X.columns), file)
print(f"Saved: {feature_names_file}")

# Test loading
print("\nTesting model loading...")
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
test_pred = loaded_model.predict(X_test[:1])
print(f"Test prediction: {test_pred[0]} ({'Placed' if test_pred[0] == 1 else 'Not Placed'})")
print("Model loaded successfully!")

print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)

print(f"\nFinal Results:")
print(f"  Best Model: {best_model}")
print(f"  Test Accuracy: {best_accuracy*100:.2f}%")
print(f"  Precision: {precision_rf*100 if best_model=='Random Forest' else precision_lr*100:.2f}%")
print(f"  Recall: {recall_rf*100 if best_model=='Random Forest' else recall_lr*100:.2f}%")
print(f"  F1-Score: {f1_rf*100 if best_model=='Random Forest' else f1_lr*100:.2f}%")

print(f"\nFiles created:")
print(f"  1. {model_filename}")
print(f"  2. {encoders_filename}")
print(f"  3. {feature_names_file}")
print(f"  4. confusion_matrices.png")
print(f"  5. model_comparison.png")

print(f"\nTo run the web app:")
print(f"  streamlit run app.py")

print("\n" + "="*80)
