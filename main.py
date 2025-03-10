import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Global parameters
epochs = 5  # Number of training epochs (iterations over the user dataset)
user_configs = [3, 5, 7, 9, 11]  # Different number of users to test
log_file = "federated_learning_logs.txt"

# Function to write logs to a file
def write_log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Fetch dataset: 20 Newsgroups (text classification dataset)
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

# Preprocessing: Convert text to numerical features (bag-of-words)
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create and train Naive Bayes model for each user
def train_user_model(X_user, y_user):
    model = MultinomialNB()
    model.fit(X_user, y_user)
    return model

# Function to perform federated learning with a specific number of users
def federated_learning(num_users):
    # Split data among the users
    user_data = np.array_split(X_train, num_users)
    user_labels = np.array_split(y_train, num_users)

    # Store models, accuracies, and weights
    user_models = []
    user_accuracies = []

    # Train models for each user
    for i in range(num_users):
        print(f"\nTraining {i + 1} / {num_users} on Naive Bayes model...")
        model = train_user_model(user_data[i], user_labels[i])
        user_preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, user_preds)

        print(f"User {i + 1} Naive Bayes Accuracy: {accuracy * 100:.2f}%")
        user_models.append(model)
        user_accuracies.append(accuracy)

    # Aggregating models (Federated Learning)
    print("\nAggregating Naive Bayes model...")

    # Initialize global model as the first user model
    global_model = user_models[0]

    # Calculate weighted average of user model predictions
    global_predictions = np.zeros((len(y_test), len(label_encoder.classes_)))
    for i in range(num_users):
        user_model = user_models[i]
        user_preds_proba = user_model.predict_proba(X_test)
        global_predictions += user_preds_proba

    # Average predictions from all user models
    global_predictions /= num_users

    # Choose the class with the highest average probability as the final prediction
    final_preds = np.argmax(global_predictions, axis=1)

    # Final accuracy of aggregated model
    final_accuracy = accuracy_score(y_test, final_preds)
    print(f"\nFinal Aggregated Naive Bayes Model Accuracy: {final_accuracy * 100:.2f}%")

    # Write logs to file
    write_log(f"\nFederated Learning with {num_users} Users:")
    for i in range(num_users):
        write_log(f"User {i + 1} Accuracy: {user_accuracies[i] * 100:.2f}%")
    write_log(f"Final Aggregated Model Accuracy: {final_accuracy * 100:.2f}")
    write_log("=" * 50)


# Run federated learning for different numbers of users
for num_users in user_configs:
    federated_learning(num_users)

print("\nTraining complete. Logs have been saved to 'federated_learning_logs.txt'.")
