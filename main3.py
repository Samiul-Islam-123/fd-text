import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

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

# Get number of classes for logistic regression
n_classes = len(np.unique(y))

def train_user_model_lr(X_user, y_user, local_epochs, learning_rate=0.1):
    """
    Train a user Logistic Regression model with local epochs.
    Uses SGD for multiple local epochs.
    """
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='saga',
        multi_class='multinomial',
        max_iter=local_epochs,
        warm_start=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_user, y_user)
    return model

def federated_learning_lr(num_users, local_epochs):
    """Federated averaging for Logistic Regression."""
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)

    user_models = []
    user_sample_sizes = []
    
    # Train local models
    for i in range(num_users):
        X_user = X_train[user_indices[i]]
        y_user = y_train[user_indices[i]]
        
        model = train_user_model_lr(X_user, y_user, local_epochs)
        user_models.append(model)
        user_sample_sizes.append(X_user.shape[0])
    
    # Federated averaging of model parameters
    total_samples = sum(user_sample_sizes)
    
    # Initialize global model parameters
    if sparse.issparse(X_train):
        n_features = X_train.shape[1]
    else:
        n_features = X_train.shape[1]
    
    # Average the coefficients and intercepts
    avg_coef = np.zeros((n_classes, n_features))
    avg_intercept = np.zeros(n_classes)
    
    for i, model in enumerate(user_models):
        weight = user_sample_sizes[i] / total_samples
        avg_coef += weight * model.coef_
        avg_intercept += weight * model.intercept_
    
    # Create global model
    global_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='saga',
        multi_class='multinomial',
        max_iter=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Manually set the parameters
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept
    global_model.classes_ = np.unique(y_train)
    global_model.n_iter_ = np.array([local_epochs])
    
    return global_model

# Run experiments
results_lr = {num_users: [] for num_users in user_configs}

for num_users in user_configs:
    for local_epochs in local_epochs_list:
        try:
            global_model = federated_learning_lr(num_users, local_epochs)
            preds = global_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results_lr[num_users].append(acc)
            print(f"[LR] Users={num_users}, Local Epochs={local_epochs}, Accuracy={acc:.4f}")
        except Exception as e:
            print(f"Error with Users={num_users}, Epochs={local_epochs}: {e}")
            results_lr[num_users].append(0)

# Plot results
plt.figure(figsize=(10, 6))
for num_users in user_configs:
    plt.plot(local_epochs_list, results_lr[num_users], marker='o', label=f"{num_users} Users")

plt.title("Accuracy vs Privacy Tradeoff (Federated Logistic Regression)")
plt.xlabel("Local Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Compare with centralized baseline
print("\nTraining centralized baseline...")
central_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='saga',
    multi_class='multinomial',
    max_iter=100,
    random_state=42,
    n_jobs=-1
)
central_model.fit(X_train, y_train)
central_acc = accuracy_score(y_test, central_model.predict(X_test))
print(f"Centralized Baseline Accuracy: {central_acc:.4f}")

# Print summary of federated results
print("\nFederated Learning Summary:")
print("Users\tBest Accuracy\tLocal Epochs")
for num_users in user_configs:
    best_acc = max(results_lr[num_users])
    best_epoch = local_epochs_list[np.argmax(results_lr[num_users])]
    print(f"{num_users}\t{best_acc:.4f}\t\t{best_epoch}")