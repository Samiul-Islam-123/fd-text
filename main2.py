import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Config
user_configs = [3, 5, 7, 9, 11]       # different privacy levels (#users)
local_epochs_list = [1, 2, 3, 5, 7]   # local computation per client

# Fetch dataset
print("Loading dataset...")
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

# Vectorize text
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_user_model_rf(X_user, y_user, local_epochs, n_trees=20):
    """
    Train a user RandomForest model with local epochs.
    Each epoch retrains a fresh forest (simulating local passes).
    The last epoch's forest is returned.
    """
    model = None
    for _ in range(local_epochs):
        model = RandomForestClassifier(n_estimators=n_trees, random_state=None, n_jobs=-1)
        model.fit(X_user, y_user)
    return model


def federated_learning_rf(num_users, local_epochs):
    """Federated averaging for Random Forest via tree aggregation."""
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)

    user_models = []
    for i in range(num_users):
        X_user = X_train[user_indices[i]]
        y_user = y_train[user_indices[i]]
        model = train_user_model_rf(X_user, y_user, local_epochs)
        user_models.append(model)

    # Create a new global model with all trees from user models
    all_estimators = []
    for model in user_models:
        all_estimators.extend(model.estimators_)
    
    # Create a new RandomForest model with all the trees
    global_model = RandomForestClassifier(n_estimators=len(all_estimators), 
                                         random_state=42, 
                                         n_jobs=-1)
    
    # Set the estimators directly
    global_model.estimators_ = all_estimators
    
    # Manually set the required attributes for prediction
    global_model.classes_ = np.unique(y_train)
    global_model.n_classes_ = len(global_model.classes_)
    
    # For RandomForest, we also need to set n_outputs_
    global_model.n_outputs_ = 1  # Single output for classification
    
    # Set the _estimator attribute (usually the base estimator)
    if len(all_estimators) > 0:
        global_model._estimator = all_estimators[0]
    
    return global_model


# Run experiments
results_rf = {num_users: [] for num_users in user_configs}

for num_users in user_configs:
    for local_epochs in local_epochs_list:
        try:
            global_model = federated_learning_rf(num_users, local_epochs)
            preds = global_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results_rf[num_users].append(acc)
            print(f"[RF] Users={num_users}, Local Epochs={local_epochs}, Accuracy={acc:.4f}")
        except Exception as e:
            print(f"Error with Users={num_users}, Epochs={local_epochs}: {e}")
            results_rf[num_users].append(0)  # Append 0 or skip

# Plot results
plt.figure(figsize=(8,6))
for num_users in user_configs:
    plt.plot(local_epochs_list, results_rf[num_users], marker='o', label=f"{num_users} Users")

plt.title("Accuracy vs Privacy Tradeoff (Federated Random Forest)")
plt.xlabel("Local Epochs (Privacy ↑)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()