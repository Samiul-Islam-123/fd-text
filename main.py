import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Config
user_configs = [3, 5, 7, 9, 11]
local_epochs_list = [1, 2, 3, 5, 7]

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

def train_user_model(X_user, y_user, local_epochs):
    """Train a user model with local_epochs using partial_fit."""
    classes = np.unique(y_train)
    model = MultinomialNB()
    for _ in range(local_epochs):
        model.partial_fit(X_user, y_user, classes=classes)
    return model

def federated_learning(num_users, local_epochs):
    """Federated averaging with local epochs."""
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)

    user_models = []
    total_samples = X_train.shape[0]
    weights = [len(user_indices[i]) / total_samples for i in range(num_users)]

    # Train user models
    for i in range(num_users):
        X_user = X_train[user_indices[i]]
        y_user = y_train[user_indices[i]]
        model = train_user_model(X_user, y_user, local_epochs)
        user_models.append(model)

    # Aggregation (FedAvg style: weighted average of counts)
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    agg_class_count = np.zeros(num_classes)
    agg_feature_count = np.zeros((num_classes, num_features))

    for i, model in enumerate(user_models):
        agg_class_count += weights[i] * model.class_count_
        agg_feature_count += weights[i] * model.feature_count_

    # Build global model
    global_model = MultinomialNB()
    global_model.classes_ = np.unique(y_train)
    global_model.class_count_ = agg_class_count
    global_model.feature_count_ = agg_feature_count
    global_model._update_class_log_prior()
    global_model._update_feature_log_prob(global_model.alpha)

    # Evaluate
    preds = global_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# Run experiments
results = {num_users: [] for num_users in user_configs}

for num_users in user_configs:
    for local_epochs in local_epochs_list:
        acc = federated_learning(num_users, local_epochs)
        results[num_users].append(acc)
        print(f"Users={num_users}, Local Epochs={local_epochs}, Accuracy={acc:.4f}")

# Plot results
plt.figure(figsize=(8,6))
for num_users in user_configs:
    plt.plot(local_epochs_list, results[num_users], marker='o', label=f"{num_users} Users")

plt.title("Accuracy vs Privacy Tradeoff (Federated Naive Bayes)")
plt.xlabel("Local Epochs (Privacy ↑)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
