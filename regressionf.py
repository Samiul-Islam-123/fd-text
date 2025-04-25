import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import joblib
import os

# Global parameters
user_configs = [3, 5, 7, 9, 11]
log_file = "federated_learning_logs_rg.txt"

# Function to write logs to a file
def write_log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Function to display model info
def model_info(model):
    print("\n=== Model Information ===")
    print(f"Classes: {model.classes_}")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_}")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Solver: {model.solver}\n")

# Function to train a user model
def train_user_model(X_user, y_user, all_classes=np.arange(20)):
    unique_classes = np.unique(y_user)
    missing_classes = np.setdiff1d(all_classes, unique_classes)
    
    if len(missing_classes) > 0:
        # Add 1 fake sample per missing class with all zeros
        n_features = X_user.shape[1]
        X_dummy = np.zeros((len(missing_classes), n_features))
        y_dummy = missing_classes
        # Stack the real and dummy data
        X_user = np.vstack([X_user, X_dummy])
        y_user = np.hstack([y_user, y_dummy])
    
    model = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    model.fit(X_user, y_user)
    return model

# Function to save model parameters
def save_model_params(model, filename):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    joblib.dump(model, filename)

# Function to extract model parameters
def extract_model_params(model):
    return {
        'coef': model.coef_,
        'intercept': model.intercept_,
        'classes': model.classes_
    }

# Function to get one sample per class
def get_one_sample_per_class(X, y, num_classes=20):
    indices = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    return X[indices], y[indices]

# Function to perform Federated Learning with FedAvg for Logistic Regression
# Modify the function to initialize global model with correct number of classes (20)
def federated_learning(num_users):
    print(f"\n{'='*50}")
    print(f"Starting Federated Learning with {num_users} users")
    print(f"{'='*50}")
    
    # Split data among users
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    user_models = []
    user_accuracies = []
    
    # Train individual models
    for i in range(num_users):
        print(f"\nTraining User {i + 1}/{num_users} on Logistic Regression model...")
        
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
        save_model_params(model, f"models/user_{i+1}_lr_model.joblib")
    
    # Aggregation Step - FedAvg for Logistic Regression
    print("\nAggregating Logistic Regression Models (FedAvg)...")
    
    # Initialize parameters for the global model
    num_classes = 20  # Set to 20 (or len(np.unique(y_train)))
    num_features = X_train.shape[1]
    
    # Create global model
    global_model = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    
    # Initialize global model with a dummy fit to set up the model correctly
    X_init, y_init = get_one_sample_per_class(X_train, y_train, num_classes=20)
    global_model.fit(X_init, y_init)
    
    # Extract the coefficient shapes for later aggregation
    global_coef = np.zeros((num_classes, num_features))
    global_intercept = np.zeros(num_classes)
    
    # Weight factors (based on data size)
    total_samples = X_train.shape[0]
    weights = [len(user_indices[i]) / total_samples for i in range(num_users)]
    
    # Weighted averaging of parameters
    for i, model in enumerate(user_models):
        if model.coef_.shape[0] == num_classes:  # Check if the number of classes matches
            global_coef += weights[i] * model.coef_
            global_intercept += weights[i] * model.intercept_
        else:
            print(f"Skipping model {i + 1} due to shape mismatch: {model.coef_.shape} vs {global_coef.shape}")
    
    # Set the aggregated parameters to the global model
    global_model.coef_ = global_coef
    global_model.intercept_ = global_intercept
    
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
    save_model_params(global_model, f"models/global_lr_model_{num_users}_users.joblib")
    
    # Logging
    write_log(f"\nFederated Learning with Logistic Regression - {num_users} Users:")
    for i in range(num_users):
        write_log(f"User {i + 1} Accuracy: {user_accuracies[i] * 100:.2f}%")
    write_log(f"Final Aggregated Model Accuracy: {global_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, global_accuracy

# Alternative FL algorithm - FedProx for Logistic Regression
def federated_learning_prox(num_users, mu=0.01):
    print(f"\n{'='*50}")
    print(f"Starting FedProx Learning with {num_users} users (mu={mu})")
    print(f"{'='*50}")
    
    # Split data among users
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    # Initialize a global model
    global_model = LogisticRegression(max_iter=100, solver='saga', random_state=42)
    global_model.fit(X_train[:100], y_train[:100])
    
    # Save initial global parameters
    global_coef = global_model.coef_.copy()
    global_intercept = global_model.intercept_.copy()
    
    # Number of communication rounds
    rounds = 3
    
    for round in range(rounds):
        print(f"\n--- Round {round+1}/{rounds} ---")
        
        user_models = []
        user_accuracies = []
        
        # Train individual models (with proximal term)
        for i in range(num_users):
            print(f"Training User {i + 1}/{num_users}...")
            
            # Extract user data using indices
            X_user = X_train[user_indices[i]]
            y_user = y_train[user_indices[i]]
            
            # For FedProx, we need to incorporate the proximal term in the objective
            # In scikit-learn, we can't directly modify the objective
            # So we'll train normally and then apply regularization towards the global model
            
            # Train model
            start_time = time.time()
            model = train_user_model(X_user, y_user)
            training_time = time.time() - start_time
            
            # Apply proximal term regularization
            if round > 0:  # Skip in the first round
                # Move model parameters closer to global parameters (proximal regularization)
                # This simulates the L2 proximal term in FedProx
                model.coef_ = (1 - mu) * model.coef_ + mu * global_coef
                model.intercept_ = (1 - mu) * model.intercept_ + mu * global_intercept
            
            # Evaluate model
            user_preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, user_preds)
            
            print(f"User {i + 1} Data Size: {X_user.shape[0]} samples")
            print(f"User {i + 1} Training Time: {training_time:.2f} seconds")
            print(f"User {i + 1} Accuracy: {accuracy * 100:.2f}%")
            
            user_models.append(model)
            user_accuracies.append(accuracy)
        
        # Aggregation Step - Average the parameters
        print("\nAggregating Models...")
        
        # Weight factors (based on data size)
        total_samples = X_train.shape[0]
        weights = [len(user_indices[i]) / total_samples for i in range(num_users)]
        
        # Reset aggregated parameters
        global_coef = np.zeros_like(global_model.coef_)
        global_intercept = np.zeros_like(global_model.intercept_)
        
        # Weighted averaging of parameters
        for i, model in enumerate(user_models):
            global_coef += weights[i] * model.coef_
            global_intercept += weights[i] * model.intercept_
        
        # Update global model
        global_model.coef_ = global_coef
        global_model.intercept_ = global_intercept
        
        # Evaluate global model
        global_preds = global_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, global_preds)
        
        print(f"Round {round+1} Global Model Accuracy: {global_accuracy * 100:.2f}%")
    
    # Final evaluation
    global_preds = global_model.predict(X_test)
    global_accuracy = accuracy_score(y_test, global_preds)
    
    print(f"\nFinal FedProx Model Accuracy: {global_accuracy * 100:.2f}%")
    print("\nClassification Report for FedProx Model:")
    print(classification_report(y_test, global_preds))
    
    # Save global model parameters
    save_model_params(global_model, f"models/global_lr_fedprox_{num_users}_users.joblib")
    
    # Logging
    write_log(f"\nFedProx Learning with Logistic Regression - {num_users} Users (mu={mu}):")
    write_log(f"Final FedProx Model Accuracy: {global_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, global_accuracy

# Classic Federated Learning - FedSGD algorithm with Logistic Regression
def federated_learning_sgd(num_users):
    print(f"\n{'='*50}")
    print(f"Starting FedSGD Learning with {num_users} users")
    print(f"{'='*50}")
    
    # Split data among users
    indices = np.arange(X_train.shape[0])
    user_indices = np.array_split(indices, num_users)
    
    # Initialize a global model
    global_model = LogisticRegression(max_iter=100, solver='saga', random_state=42)
    global_model.fit(X_train[:100], y_train[:100])
    
    # Save initial global parameters
    global_coef = global_model.coef_.copy()
    global_intercept = global_model.intercept_.copy()
    
    # Number of communication rounds
    rounds = 3
    best_accuracy = 0
    best_model = None
    
    for round in range(rounds):
        print(f"\n--- Round {round+1}/{rounds} ---")
        
        user_models = []
        user_accuracies = []
        
        # Calculate gradients (difference from global model)
        coef_updates = []
        intercept_updates = []
        
        # Train individual models
        for i in range(num_users):
            print(f"Training User {i + 1}/{num_users}...")
            
            # Extract user data using indices
            X_user = X_train[user_indices[i]]
            y_user = y_train[user_indices[i]]
            
            # Clone the current global model
            model = LogisticRegression(max_iter=100, solver='saga', random_state=42)
            model.fit(X_user[:10], y_user[:10])  # Just to initialize
            
            # Set parameters to current global values
            model.coef_ = global_coef.copy()
            model.intercept_ = global_intercept.copy()
            
            # Train model 
            start_time = time.time()
            model = train_user_model(X_user, y_user)
            training_time = time.time() - start_time
            
            # Calculate update (difference from global model)
            coef_update = model.coef_ - global_coef
            intercept_update = model.intercept_ - global_intercept
            
            # Store updates
            coef_updates.append(coef_update)
            intercept_updates.append(intercept_update)
            
            # Evaluate model
            user_preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, user_preds)
            
            print(f"User {i + 1} Data Size: {X_user.shape[0]} samples")
            print(f"User {i + 1} Training Time: {training_time:.2f} seconds")
            print(f"User {i + 1} Accuracy: {accuracy * 100:.2f}%")
            
            user_models.append(model)
            user_accuracies.append(accuracy)
        
        # Aggregation Step - Apply average updates to global model
        print("\nAggregating Client Updates...")
        
        # Weight factors (based on data size)
        total_samples = X_train.shape[0]
        weights = [len(user_indices[i]) / total_samples for i in range(num_users)]
        
        # Apply weighted updates to global model
        for i in range(num_users):
            global_coef += weights[i] * coef_updates[i]
            global_intercept += weights[i] * intercept_updates[i]
        
        # Update global model parameters
        global_model.coef_ = global_coef
        global_model.intercept_ = global_intercept
        
        # Evaluate global model
        global_preds = global_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, global_preds)
        
        print(f"Round {round+1} Global Model Accuracy: {global_accuracy * 100:.2f}%")
        
        # Save best model
        if global_accuracy > best_accuracy:
            best_accuracy = global_accuracy
            best_model = global_model
            save_model_params(global_model, f"models/global_lr_fedsgd_{num_users}_users_best.joblib")
    
    # Final evaluation
    global_preds = global_model.predict(X_test)
    global_accuracy = accuracy_score(y_test, global_preds)
    
    print(f"\nFinal FedSGD Model Accuracy: {global_accuracy * 100:.2f}%")
    print("\nClassification Report for FedSGD Model:")
    print(classification_report(y_test, global_preds))
    
    # Display global model parameters
    print("\nGlobal Model Parameters:")
    model_info(global_model)
    
    # Save final model parameters
    save_model_params(global_model, f"models/global_lr_fedsgd_{num_users}_users.joblib")
    
    # Logging
    write_log(f"\nClassic FedSGD Learning with Logistic Regression - {num_users} Users:")
    for i in range(num_users):
        write_log(f"User {i + 1} Accuracy: {user_accuracies[i] * 100:.2f}%")
    write_log(f"Final FedSGD Model Accuracy: {global_accuracy * 100:.2f}%")
    write_log("=" * 50)
    
    return global_model, global_accuracy

# Main execution
if __name__ == "__main__":
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write("Federated Learning Experiments with Logistic Regression\n")
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
    print("\nRunning FedSGD algorithm...")
    for num_users in user_configs:
        federated_learning_sgd(num_users)
    
    print("\nTraining complete. Logs saved to 'federated_learning_logs_rg.txt'.")