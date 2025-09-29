
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import joblib 

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, adjusted_rand_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline 
from ucimlrepo import fetch_ucirepo



heart_disease = fetch_ucirepo(id=45)


X = heart_disease.data.features
y = heart_disease.data.targets


df = pd.concat([X, y], axis=1)
print("Initial DataFrame shape:", df.shape)


for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)


df_encoded = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df_encoded.shape)


X_features = df_encoded.drop(columns=['num']) 

y_target = (df_encoded['num'] > 0).astype(int)


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)
print("Scaled Features Shape:", X_scaled.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_target, test_size=0.2, random_state=42, stratify=y_target
)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_importances = pd.Series(rf.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)



plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances.values, y=rf_importances.index, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("\nTop 10 Features (Random Forest):")
print(rf_importances.head(10))


xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

xgb_importances = pd.Series(xgb.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances.values, y=xgb_importances.index, palette="coolwarm")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("\nTop 10 Features (XGBoost):")
print(xgb_importances.head(10))



rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=8)
rfe_selector.fit(X_train, y_train)

rfe_selected_features = X_scaled.columns[rfe_selector.support_]
print("\nSelected Features using RFE:")
print(list(rfe_selected_features))


rfe_ranking = pd.Series(rfe_selector.ranking_, index=X_scaled.columns).sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=rfe_ranking.values, y=rfe_ranking.index, palette="plasma")
plt.title("RFE Feature Ranking (Lower = Better)")
plt.xlabel("Rank")
plt.ylabel("Features")
plt.show()


X_chi = X_scaled - X_scaled.min().min()

chi_selector = SelectKBest(score_func=chi2, k=8)
chi_selector.fit(X_chi, y_target)

chi_scores = pd.Series(chi_selector.scores_, index=X_scaled.columns).sort_values(ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x=chi_scores.values, y=chi_scores.index, palette="mako")
plt.title("Chi-Square Test Feature Scores")
plt.xlabel("Chi-Square Score")
plt.ylabel("Features")
plt.show()

print("\nTop 8 Features (Chi-Square Test):")
print(chi_scores.head(8))


top_rf = set(rf_importances.head(8).index)
top_xgb = set(xgb_importances.head(8).index)
top_rfe = set(rfe_selected_features)
top_chi = set(chi_scores.head(8).index)

final_features = list((top_rf & top_xgb) | (top_rfe & top_chi))
print("\nFinal Selected Features for Modeling:")
print(final_features)

X_final = X_scaled[final_features]
print("\nFinal Dataset Shape:", X_final.shape)


plt.figure(figsize=(8, 6))
sns.heatmap(X_final.corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap of Final Selected Features")
plt.show()


X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y_target, test_size=0.2, random_state=42, stratify=y_target
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []

print("\n" + "="*50)
print("Step 10: Supervised Model Performance")
print("="*50)

plt.figure(figsize=(8, 6))

for name, model in models.items():
    
    model.fit(X_train_final, y_train_final)
    
   
    y_pred = model.predict(X_test_final)
    y_pred_proba = model.predict_proba(X_test_final)[:, 1]
    
   
    acc = accuracy_score(y_test_final, y_pred)
    prec = precision_score(y_test_final, y_pred) 
    rec = recall_score(y_test_final, y_pred) 
    f1 = f1_score(y_test_final, y_pred) 
    auc = roc_auc_score(y_test_final, y_pred_proba) 
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": auc
    })
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")


plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()


results_df = pd.DataFrame(results)
print("\nModel Performance Summary (Supervised Classification):\n")
print(results_df.sort_values(by='AUC', ascending=False))

# Detailed Classification Reports
for name, model in models.items():
    print(f"\nClassification Report for {name}:\n")
    y_pred = model.predict(X_test_final)
    print(classification_report(y_test_final, y_pred))



print("\n" + "="*50)
print("Step 13: Unsupervised Clustering Analysis")
print("="*50)

# --- 13.1 K-Means: Elbow Method to Determine K ---
print("\nK-Means Clustering: Determining Optimal K (Elbow Method)")
ssd = [] # Sum of Squared Distances
K_range = range(1, 11)

for k in K_range:
    # Set n_init to 'auto' to avoid the FutureWarning
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
    kmeans.fit(X_final)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, ssd, marker='o', linestyle='--')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.grid(True)
plt.show()

# Choose K=2, matching the binary target structure
optimal_k = 2 
print(f"Chosen optimal K for K-Means: {optimal_k}")

# --- 13.2 K-Means: Apply Clustering ---
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto', max_iter=300)
kmeans_labels = kmeans_model.fit_predict(X_final)


# --- 13.3 Hierarchical Clustering: Dendrogram Analysis ---
print("\nHierarchical Clustering: Generating Dendrogram")
linked = linkage(X_final, method='ward') 

plt.figure(figsize=(15, 7))
dendrogram(
    linked,
    orientation='top',
    # Note: Using y_target for labels here is purely for comparison, not part of clustering logic
    distance_sort='descending',
    show_leaf_counts=True
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')
plt.show()



hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical_model.fit_predict(X_final)



print("\n" + "="*50)
print("Step 14: Cluster Comparison with Actual Binarized Labels")
print("="*50)


ari_kmeans = adjusted_rand_score(y_target, kmeans_labels)
print(f"K-Means (K={optimal_k}) vs Actual Labels (ARI): {ari_kmeans:.4f}")

ari_hierarchical = adjusted_rand_score(y_target, hierarchical_labels)
print(f"Hierarchical Clustering (K={optimal_k}) vs Actual Labels (ARI): {ari_hierarchical:.4f}")


cluster_df = X_final.copy()
cluster_df['KMeans_Cluster'] = kmeans_labels
cluster_df['Actual_Label'] = y_target.values


print("\n" + "="*50)
print("Step 15: Saving the Best Model and Pipeline")
print("="*50)


best_row = results_df.loc[results_df['AUC'].idxmax()]
best_model_name = best_row['Model']
best_model_auc = best_row['AUC']
model_obj = models[best_model_name]

print(f"Overall Best Model: {best_model_name} (AUC: {best_model_auc:.3f})")


class FeatureSelector(object):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        # Identify the indices of the final features in the original X_features columns
        self.indices_ = [X.columns.get_loc(col) for col in self.feature_names]
        return self
    def transform(self, X):
        return X.iloc[:, self.indices_] # Select columns by index


full_pipeline = Pipeline(steps=[
    ('feature_selector', FeatureSelector(final_features)),
    ('scaler', StandardScaler()),
    ('classifier', model_obj)
])


print("Training final full pipeline on all data...")
full_pipeline.fit(X_features, y_target)


model_filename = 'heart_disease_predictor_pipeline.pkl'
joblipipelb.dump(full_ine, model_filename)

print(f"\nSuccessfully saved the full model pipeline to: {model_filename}")
print("To load and predict: joblib.load('heart_disease_predictor_pipeline.pkl')")

print("\nFinal Analysis and Model Saving Complete.")

