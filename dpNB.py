import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Config - Focus on meaningful epsilon range (0.1 to 5.0)
privacy_mechanisms = ['none', 'laplace', 'gaussian', 'staircase']
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Only meaningful privacy budgets

# Fetch and preprocess data
print("Loading dataset...")
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

n_classes = len(np.unique(y_encoded))
n_features = X_vectorized.shape[1]

print(f"Data shape: {X_train.shape}")
print(f"Classes: {n_classes}, Features: {n_features}")

# Differential Privacy Mechanisms
def laplace_mechanism(data, epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

def gaussian_mechanism(data, epsilon, delta, sensitivity):
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def staircase_mechanism(data, epsilon, sensitivity, gamma=0.7):
    lambda_val = (np.exp(epsilon) - 1) / (np.exp(epsilon) + 1)
    scale = sensitivity / epsilon
    
    noise = np.zeros_like(data)
    flat_data = data.flatten()
    
    for i in range(flat_data.size):
        u = np.random.uniform(0, 1)
        if u < gamma:
            k = np.random.geometric(1 - np.exp(-epsilon))
            sign = 1 if np.random.random() < 0.5 else -1
            noise.flat[i] = sign * k * scale
        else:
            k = np.random.geometric(1 - np.exp(-epsilon/lambda_val))
            sign = 1 if np.random.random() < 0.5 else -1
            noise.flat[i] = sign * k * scale * lambda_val
    
    return data + noise

def apply_privacy_mechanism(data, mechanism, epsilon, delta=1e-5, sensitivity=1.0):
    if mechanism == 'none':
        return data
    if mechanism == 'laplace':
        return laplace_mechanism(data, epsilon, sensitivity)
    elif mechanism == 'gaussian':
        return gaussian_mechanism(data, epsilon, delta, sensitivity)
    elif mechanism == 'staircase':
        return staircase_mechanism(data, epsilon, sensitivity)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

class DifferentialPrivateNaiveBayes:
    def __init__(self, mechanism='none', epsilon=1.0, alpha=1.0):
        self.mechanism = mechanism
        self.epsilon = epsilon
        self.alpha = alpha
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Count class occurrences
        self.class_count_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == c)
        
        # Count feature occurrences per class
        self.feature_count_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            class_mask = (y == c)
            if np.any(class_mask):
                self.feature_count_[i, :] = X[class_mask].sum(axis=0).A1
        
        # Apply Differential Privacy to feature counts
        if self.mechanism != 'none':
            sensitivity = 1.0
            self.feature_count_ = apply_privacy_mechanism(
                self.feature_count_, self.mechanism, self.epsilon, sensitivity=sensitivity
            )
            self.feature_count_ = np.maximum(self.feature_count_, 0)
        
        # Calculate probabilities
        self.class_prior_ = self.class_count_ / np.sum(self.class_count_)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            numerator = self.feature_count_[i, :] + self.alpha
            denominator = self.class_count_[i] + self.alpha * n_features
            self.feature_log_prob_[i, :] = np.log(numerator / denominator)
        
        return self
    
    def predict(self, X):
        if self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet.")
        
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            if sparse.issparse(X):
                class_log_prob = self.feature_log_prob_[i, :]
                log_probs[:, i] = X.dot(class_log_prob) + np.log(self.class_prior_[i])
            else:
                log_probs[:, i] = np.sum(X * self.feature_log_prob_[i, :], axis=1) + np.log(self.class_prior_[i])
        
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

# Experiment: Test DP Naive Bayes
results_nb = {}

for mechanism in privacy_mechanisms:
    print(f"\n{'='*50}")
    print(f"Testing {mechanism.upper()} mechanism")
    print(f"{'='*50}")
    
    results_nb[mechanism] = {}
    
    for epsilon in epsilon_values:
        print(f"ε={epsilon}...", end=" ")
        
        dp_nb = DifferentialPrivateNaiveBayes(mechanism=mechanism, epsilon=epsilon, alpha=1.0)
        dp_nb.fit(X_train, y_train)
        accuracy = dp_nb.score(X_test, y_test)
        results_nb[mechanism][epsilon] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")

# Compare with regular Naive Bayes
regular_nb = MultinomialNB(alpha=1.0)
regular_nb.fit(X_train, y_train)
regular_accuracy = regular_nb.score(X_test, y_test)
print(f"\nRegular Naive Bayes Accuracy: {regular_accuracy:.4f}")

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy vs Epsilon (main plot)
colors = {'laplace': 'blue', 'gaussian': 'red', 'staircase': 'green', 'none': 'black'}
markers = {'laplace': 'o', 'gaussian': 's', 'staircase': '^', 'none': 'x'}

for mechanism in privacy_mechanisms:
    accuracies = [results_nb[mechanism][eps] for eps in epsilon_values]
    
    if mechanism == 'none':
        ax1.axhline(y=accuracies[0], color=colors[mechanism], linestyle='--', 
                   label='No DP', linewidth=2, alpha=0.7)
    else:
        ax1.plot(epsilon_values, accuracies, marker=markers[mechanism], 
                color=colors[mechanism], linewidth=2, markersize=8,
                label=mechanism.capitalize())

ax1.axhline(y=regular_accuracy, color='purple', linestyle=':', 
           label='Regular NB', linewidth=2)
ax1.set_xlabel('Epsilon (ε) - Privacy Budget')
ax1.set_ylabel('Accuracy')
ax1.set_title('Naive Bayes: Accuracy vs Privacy Budget')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Privacy-Utility Tradeoff (Accuracy Drop)
baseline_acc = results_nb['none'][1.0]

for mechanism in [m for m in privacy_mechanisms if m != 'none']:
    accuracy_drops = [baseline_acc - results_nb[mechanism][eps] for eps in epsilon_values]
    ax2.plot(epsilon_values, accuracy_drops, marker=markers[mechanism], 
            color=colors[mechanism], linewidth=2, markersize=8,
            label=mechanism.capitalize())

ax2.set_xlabel('Epsilon (ε) - Privacy Budget')
ax2.set_ylabel('Accuracy Drop from Baseline')
ax2.set_title('Privacy Cost: Accuracy Loss vs Privacy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Plot 3: Mechanism Comparison at each Epsilon
epsilon_labels = [f'ε={eps}' for eps in epsilon_values]
x_pos = np.arange(len(epsilon_values))
width = 0.25

for i, mechanism in enumerate([m for m in privacy_mechanisms if m != 'none']):
    accuracies = [results_nb[mechanism][eps] for eps in epsilon_values]
    ax3.bar(x_pos + i*width, accuracies, width, label=mechanism.capitalize(),
           color=colors[mechanism], alpha=0.8)

ax3.set_xlabel('Privacy Budget')
ax3.set_ylabel('Accuracy')
ax3.set_title('Mechanism Comparison at Different Privacy Levels')
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(epsilon_labels)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Signal-to-Noise Ratio (Conceptual)
# For visualization purposes - showing relative noise levels
conceptual_snr = {
    'laplace': [0.1, 0.5, 1.0, 2.0, 4.0],
    'gaussian': [0.05, 0.3, 0.6, 1.2, 2.5],
    'staircase': [0.2, 1.0, 2.0, 4.0, 8.0]
}

for mechanism in ['laplace', 'gaussian', 'staircase']:
    ax4.plot(epsilon_values, conceptual_snr[mechanism], marker=markers[mechanism],
            color=colors[mechanism], linewidth=2, markersize=8,
            label=mechanism.capitalize())

ax4.set_xlabel('Epsilon (ε) - Privacy Budget')
ax4.set_ylabel('Conceptual Signal-to-Noise Ratio')
ax4.set_title('Relative Noise Levels of Different Mechanisms')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')
ax4.set_yscale('log')

plt.tight_layout()
plt.show()

# Print detailed results table
print("\n" + "="*80)
print("NAIVE BAYES WITH DIFFERENTIAL PRIVACY - RESULTS SUMMARY")
print("="*80)
print(f"{'Mechanism':<12} {'ε=0.1':<8} {'ε=0.5':<8} {'ε=1.0':<8} {'ε=2.0':<8} {'ε=5.0':<8}")

for mechanism in privacy_mechanisms:
    print(f"{mechanism.upper():<12}", end="")
    for epsilon in epsilon_values:
        acc = results_nb[mechanism][epsilon]
        print(f" {acc:.4f}  ", end="")
    print()

print(f"\n{'Regular NB':<12} {regular_accuracy:.4f}")

# Print privacy-utility analysis
print(f"\n{'='*60}")
print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
print(f"{'='*60}")
print("Lower ε = More Privacy = More Noise = Lower Accuracy")
print("Higher ε = Less Privacy = Less Noise = Higher Accuracy")
print("\nRecommended ε values:")
print("ε=0.1  : Strong privacy     (significant accuracy loss)")
print("ε=1.0  : Moderate privacy   (reasonable balance)")
print("ε=5.0  : Weak privacy       (minimal accuracy loss)")

# Show best mechanism recommendations
print(f"\n{'='*60}")
print("BEST MECHANISM RECOMMENDATIONS")
print(f"{'='*60}")

for epsilon in epsilon_values:
    best_mechanism = None
    best_accuracy = 0
    for mechanism in ['laplace', 'gaussian', 'staircase']:
        if results_nb[mechanism][epsilon] > best_accuracy:
            best_accuracy = results_nb[mechanism][epsilon]
            best_mechanism = mechanism
    
    print(f"At ε={epsilon}: {best_mechanism.upper()} wins with accuracy {best_accuracy:.4f}")