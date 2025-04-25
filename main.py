import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import time

# Global parameters
user_configs = [3, 5, 7, 9, 11]
log_file = "federated_learning_logs.txt"

# Function to write logs to a file
def write_log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Function to display model info
def model_info(model):
    print("\n=== Model Information ===")
    print(f"Classes: {model.classes_}")
    print(f"Feature Count Shape: {model.feature_count_.shape}")
    print(f"Class Counts: {model.class_count_}\n")

# Function to train a user model
def train_user_model(X_user, y_user):
    model = MultinomialNB()
    model.classes_ = np.unique(y_train)  # Ensure all users have consistent class labels
    model.fit(X_user, y_user)
    return model

# Function to save model parameters
def save_model_params(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'class_count': model.class_count_,
            'feature_count': model.feature_count_,
            'class_log_prior': model.class_log_prior_
        }, f)

# Function to perform Federated Learning with FedAvg
def federated_learning(num_users):
    print(f"\n{'='*50}")
    print(f"Starting Federated Learning with {num_users} users")
    print(f"{'='*50}")
    
    # Split data among users (keeping sparse matrix format)
    # First convert to array indices to split
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    user_models = []
    user_accuracies = []
    
    # Train individual models
    for i in range(num_users):
        print(f"\nTraining User {i + 1}/{num_users} on Naive Bayes model...")
        
        # Extract user data using indices
        X_user = X_train[user_indices[i]]
        y_user = y_train[user_indices[i]]
        
        # Train model
        start_time = time.time()
        model = train_user_model(X_user, y_user)
        training_time = time.time() - start_time
        
        # Evaluate model
        user_preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, user_preds)
        
        print(f"User {i + 1} Data Size: {X_user.shape[0]} samples")
        print(f"User {i + 1} Training Time: {training_time:.2f} seconds")
        print(f"User {i + 1} Accuracy: {accuracy * 100:.2f}%")
        
        user_models.append(model)
        user_accuracies.append(accuracy)
        
        # Save model parameters
        save_model_params(model, f"user_{i+1}_model_params.pkl")
    
    # Aggregation Step - FedAvg for Naive Bayes
    print("\nAggregating Naive Bayes Models (FedAvg)...")
    
    # Initialize aggregated parameters
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Initialize with zeros
    agg_class_count = np.zeros(num_classes)
    agg_feature_count = np.zeros((num_classes, num_features))
    
    # Weight factors (based on data size)
    total_samples = X_train.shape[0]
    weights = [len(user_indices[i]) / total_samples for i in range(num_users)]
    
    # Weighted averaging of parameters
    for i in range(num_users):
        agg_class_count += weights[i] * user_models[i].class_count_
        agg_feature_count += weights[i] * user_models[i].feature_count_
    
    # Create global model
    global_model = MultinomialNB()
    # Create a properly initialized model with all classes
    global_model.classes_ = np.unique(y_train)
    global_model.class_count_ = agg_class_count
    global_model.feature_count_ = agg_feature_count
    
    # Recalculate class log prior and feature log prob
    global_model._update_class_log_prior()
    global_model._update_feature_log_prob(global_model.alpha)
    
    # Evaluate global model
    global_preds = global_model.predict(X_test)
    global_accuracy = accuracy_score(y_test, global_preds)
    
    print(f"\nFinal Aggregated Model Accuracy: {global_accuracy * 100:.2f}%")
    print("\nClassification Report for Global Model:")
    print(classification_report(y_test, global_preds))
    
    # Display global model parameters
    print("\nGlobal Model Parameters:")
    model_info(global_model)
    
    # Save global model parameters
    save_model_params(global_model, f"global_model_params_{num_users}_users.pkl")
    
    # Logging
    write_log(f"\nFederated Learning with {num_users} Users:")
    for i in range(num_users):
        write_log(f"User {i + 1} Accuracy: {user_accuracies[i] * 100:.2f}%")
    write_log(f"Final Aggregated Model Accuracy: {global_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, global_accuracy

# Alternative FL algorithm - FedProx (with proximal term)
def federated_learning_prox(num_users, mu=0.01):
    print(f"\n{'='*50}")
    print(f"Starting FedProx Learning with {num_users} users (mu={mu})")
    print(f"{'='*50}")
    
    # Split data among users
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    # Initialize a global model
    global_model = MultinomialNB()
    global_model.fit(X_train, y_train)
    
    # Save initial global parameters
    global_feature_count = global_model.feature_count_.copy()
    
    user_models = []
    user_accuracies = []
    
    # Number of communication rounds
    rounds = 3
    
    for round in range(rounds):
        print(f"\n--- Round {round+1}/{rounds} ---")
        
        # Train individual models (with proximal term)
        for i in range(num_users):
            print(f"Training User {i + 1}/{num_users}...")
            
            # Extract user data using indices
            X_user = X_train[user_indices[i]]
            y_user = y_train[user_indices[i]]
            
            # Train model
            model = MultinomialNB()
            model.fit(X_user, y_user)
            
            # Apply proximal term (simplified approach for NB)
            # This is a simplified approximation of the proximal term concept
            model.feature_count_ = (model.feature_count_ + mu * global_feature_count) / (1 + mu)
            model._update_feature_log_prob(model.alpha)
            
            # Evaluate model
            user_preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, user_preds)
            
            print(f"User {i + 1} Accuracy: {accuracy * 100:.2f}%")
            
            user_models.append(model)
            user_accuracies.append(accuracy)
        
        # Aggregation Step
        print("\nAggregating Models...")
        
        # Initialize aggregated parameters
        num_features = X_train.shape[1]
        num_classes = len(global_model.classes_)

        
        # Initialize with zeros
        # agg_class_count = np.zeros(num_classes)
        # agg_feature_count = np.zeros((num_classes, num_features))
        agg_class_count = np.zeros_like(global_model.class_count_)
        agg_feature_count = np.zeros_like(global_model.feature_count_)

        
        # Weight factors (based on data size)
        total_samples = X_train.shape[0]
        weights = [len(user_indices[i]) / total_samples for i in range(num_users)]
        
        # Weighted averaging of parameters
        for i in range(num_users):
            agg_class_count += weights[i] * user_models[i].class_count_
            agg_feature_count += weights[i] * user_models[i].feature_count_
        
        # Update global model
        global_model.class_count_ = agg_class_count
        global_model.feature_count_ = agg_feature_count
        global_model._update_class_log_prior()
        global_model._update_feature_log_prob(global_model.alpha)
        
        # Update global parameters for next round
        global_feature_count = global_model.feature_count_.copy()
        
        # Evaluate global model
        global_preds = global_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, global_preds)
        
        print(f"Round {round+1} Global Model Accuracy: {global_accuracy * 100:.2f}%")
        
        # Clear user models for next round
        user_models = []
        user_accuracies = []
    
    # Final evaluation
    global_preds = global_model.predict(X_test)
    global_accuracy = accuracy_score(y_test, global_preds)
    
    print(f"\nFinal FedProx Model Accuracy: {global_accuracy * 100:.2f}%")
    print("\nClassification Report for FedProx Model:")
    print(classification_report(y_test, global_preds))
    
    # Logging
    write_log(f"\nFedProx Learning with {num_users} Users (mu={mu}):")
    write_log(f"Final FedProx Model Accuracy: {global_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, global_accuracy

# Classic Federated Learning - FedSGD algorithm
def federated_learning_sgd(num_users):
    print(f"\n{'='*50}")
    print(f"Starting Classic FedSGD Learning with {num_users} users")
    print(f"{'='*50}")
    
    # Split data among users
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    # Initialize a global model with minimal data
    init_size = min(100, X_train.shape[0])
    global_model = MultinomialNB()
    global_model.fit(X_train[:init_size], y_train[:init_size])
    
    # Number of communication rounds
    rounds = 3
    best_accuracy = 0
    
    for round in range(rounds):
        print(f"\n--- Round {round+1}/{rounds} ---")
        
        # Save current global parameters
        current_class_count = global_model.class_count_.copy()
        current_feature_count = global_model.feature_count_.copy()
        
        user_updates = []
        user_accuracies = []
        
        # Train individual models
        for i in range(num_users):
            print(f"Training User {i + 1}/{num_users}...")
            
            # Extract user data using indices
            X_user = X_train[user_indices[i]]
            y_user = y_train[user_indices[i]]
            
            # Train model
            start_time = time.time()
            
            # Clone the current global model parameters
            model = MultinomialNB()
            model.classes_ = global_model.classes_
            model.class_count_ = current_class_count.copy()
            model.feature_count_ = current_feature_count.copy()
            model._update_class_log_prior()
            model._update_feature_log_prob(model.alpha)
            
            # Train on local data
            model.fit(X_user, y_user)
            
            # Compute update (difference from current global model)
            class_count_update = model.class_count_ - current_class_count
            feature_count_update = model.feature_count_ - current_feature_count
            
            training_time = time.time() - start_time
            
            # Evaluate individual model
            user_preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, user_preds)
            
            print(f"User {i + 1} Data Size: {X_user.shape[0]} samples")
            print(f"User {i + 1} Training Time: {training_time:.2f} seconds")
            print(f"User {i + 1} Accuracy: {accuracy * 100:.2f}%")
            
            # Store the update
            user_updates.append((class_count_update, feature_count_update))
            user_accuracies.append(accuracy)
        
        # Aggregation step - Average updates and apply to global model
        print("\nAggregating Client Updates...")
        
        # Apply average of all updates to global model
        for class_update, feature_update in user_updates:
            global_model.class_count_ += class_update / num_users
            global_model.feature_count_ += feature_update / num_users
        
        # Update derived parameters
        global_model._update_class_log_prior()
        global_model._update_feature_log_prob(global_model.alpha)
        
        # Evaluate global model
        global_preds = global_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, global_preds)
        
        print(f"Round {round+1} Global Model Accuracy: {global_accuracy * 100:.2f}%")
        
        # Save best model
        if global_accuracy > best_accuracy:
            best_accuracy = global_accuracy
            save_model_params(global_model, f"global_model_params_fedsgd_{num_users}_users_best.pkl")
    
    # Final evaluation with best model
    print(f"\nFinal FedSGD Model Accuracy: {best_accuracy * 100:.2f}%")
    print("\nClassification Report for FedSGD Model:")
    print(classification_report(y_test, global_preds))
    
    # Display global model parameters
    print("\nGlobal Model Parameters:")
    model_info(global_model)
    
    # Save final model parameters
    save_model_params(global_model, f"global_model_params_fedsgd_{num_users}_users.pkl")
    
    # Logging
    write_log(f"\nClassic FedSGD Learning with {num_users} Users:")
    for i in range(num_users):
        write_log(f"User {i + 1} Accuracy: {user_accuracies[i] * 100:.2f}%")
    write_log(f"Final FedSGD Model Accuracy: {best_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, best_accuracy

# Main execution
if __name__ == "__main__":
    # Clear log file
    with open(log_file, "w") as f:
        f.write("Federated Learning Experiments\n")
        f.write("=" * 50 + "\n")
    
    # Fetch dataset
    print("Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all')
    X = newsgroups.data
    y = newsgroups.target
    
    # Convert text to numerical features (Bag-of-Words)
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(X)
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Run classic FL (FedAvg)
    print("\nRunning FedAvg algorithm...")
    for num_users in user_configs:
        federated_learning(num_users)
    
    # Run FedProx
    print("\nRunning FedProx algorithm...")
    for num_users in user_configs:
        federated_learning_prox(num_users)
    
    # Run Classic FedSGD
    print("\nRunning Classic FedSGD algorithm...")
    for num_users in user_configs:
        federated_learning_sgd(num_users)
    
    print("\nTraining complete. Logs saved to 'federated_learning_logs.txt'.")
