import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Config
privacy_mechanisms = ['none', 'laplace', 'gaussian', 'staircase']
epsilon_values = [0.1, 0.5, 1.0, 2.0]

# Fetch dataset
print("Loading dataset...")
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

# Vectorize text
vectorizer = CountVectorizer(stop_words='english', max_features=500)
X_vectorized = vectorizer.fit_transform(X)

print(f"Original data shape: {X_vectorized.shape}")
print(f"Feature value range: [{X_vectorized.data.min()}, {X_vectorized.data.max()}]")
print(f"Sparsity: {1 - (X_vectorized.nnz / (X_vectorized.shape[0] * X_vectorized.shape[1])):.3f}")

# Differential Privacy Mechanisms for DATA
def laplace_mechanism_data(data, epsilon, sensitivity):
    """Apply Laplace noise to data features."""
    if sparse.issparse(data):
        data = data.toarray()
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return np.maximum(data + noise, 0)

def gaussian_mechanism_data(data, epsilon, delta, sensitivity):
    """Apply Gaussian noise to data features."""
    if sparse.issparse(data):
        data = data.toarray()
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return np.maximum(data + noise, 0)

def staircase_mechanism_data(data, epsilon, sensitivity, gamma=0.7):
    """Apply Staircase noise to data features."""
    if sparse.issparse(data):
        data = data.toarray()
    
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
    
    return np.maximum(data + noise, 0)

def apply_dp_to_data(X, mechanism, epsilon, delta=1e-5):
    """Apply DP directly to the feature data."""
    sensitivity = 1.0  # For count data
    
    if mechanism == 'none':
        return X
    
    if mechanism == 'laplace':
        return laplace_mechanism_data(X, epsilon, sensitivity)
    
    elif mechanism == 'gaussian':
        return gaussian_mechanism_data(X, epsilon, delta, sensitivity)
    
    elif mechanism == 'staircase':
        return staircase_mechanism_data(X, epsilon, sensitivity)
    
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

def calculate_data_metrics(original, noisy, mechanism, epsilon):
    """Calculate various metrics to compare original and noisy data."""
    if sparse.issparse(original):
        original = original.toarray()
    if sparse.issparse(noisy):
        noisy = noisy.toarray()
    
    # Flatten for calculations
    orig_flat = original.flatten()
    noisy_flat = noisy.flatten()
    
    metrics = {
        'mechanism': mechanism,
        'epsilon': epsilon,
        'mse': mean_squared_error(orig_flat, noisy_flat),
        'mae': mean_absolute_error(orig_flat, noisy_flat),
        'rmse': np.sqrt(mean_squared_error(orig_flat, noisy_flat)),
        'noise_std': np.std(noisy_flat - orig_flat),
        'signal_std': np.std(orig_flat),
        'snr': np.std(orig_flat) / np.std(noisy_flat - orig_flat) if np.std(noisy_flat - orig_flat) > 0 else float('inf')
    }
    
    # Correlation metrics (on non-zero elements for sparse data)
    non_zero_mask = (orig_flat > 0) | (noisy_flat > 0)
    if np.sum(non_zero_mask) > 10:  # Need enough samples
        metrics['pearson'] = pearsonr(orig_flat[non_zero_mask], noisy_flat[non_zero_mask])[0]
        metrics['spearman'] = spearmanr(orig_flat[non_zero_mask], noisy_flat[non_zero_mask])[0]
    else:
        metrics['pearson'] = 0
        metrics['spearman'] = 0
    
    return metrics

# Experiment: Apply DP to data and compare metrics
results = []

# Sample a subset for faster computation
sample_size = min(1000, X_vectorized.shape[0])
sample_indices = np.random.choice(X_vectorized.shape[0], sample_size, replace=False)
X_sample = X_vectorized[sample_indices]

print(f"\nAnalyzing {sample_size} documents...")

for mechanism in privacy_mechanisms:
    print(f"\n{'='*50}")
    print(f"Testing {mechanism.upper()} mechanism")
    print(f"{'='*50}")
    
    for epsilon in epsilon_values:
        print(f"Epsilon = {epsilon}...", end=" ")
        
        if mechanism == 'none':
            X_noisy = X_sample
        else:
            X_noisy = apply_dp_to_data(X_sample, mechanism, epsilon)
        
        metrics = calculate_data_metrics(X_sample, X_noisy, mechanism, epsilon)
        results.append(metrics)
        
        print(f"MSE: {metrics['mse']:.3f}, SNR: {metrics['snr']:.3f}")

# Convert results to DataFrame for easier analysis
import pandas as pd
df_results = pd.DataFrame(results)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot 1: MSE vs Epsilon
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[0].plot(df_results[mask]['epsilon'], df_results[mask]['mse'], 
                    marker='o', label=mechanism.capitalize())
axes[0].set_xscale('log')
axes[0].set_xlabel('Epsilon')
axes[0].set_ylabel('Mean Squared Error')
axes[0].set_title('MSE vs Privacy Budget')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: SNR vs Epsilon
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[1].plot(df_results[mask]['epsilon'], df_results[mask]['snr'], 
                    marker='s', label=mechanism.capitalize())
axes[1].set_xscale('log')
axes[1].set_xlabel('Epsilon')
axes[1].set_ylabel('Signal-to-Noise Ratio')
axes[1].set_title('SNR vs Privacy Budget')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Pearson Correlation vs Epsilon
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[2].plot(df_results[mask]['epsilon'], df_results[mask]['pearson'], 
                    marker='^', label=mechanism.capitalize())
axes[2].set_xscale('log')
axes[2].set_xlabel('Epsilon')
axes[2].set_ylabel('Pearson Correlation')
axes[2].set_title('Correlation vs Privacy Budget')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Noise Standard Deviation
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[3].plot(df_results[mask]['epsilon'], df_results[mask]['noise_std'], 
                    marker='d', label=mechanism.capitalize())
axes[3].set_xscale('log')
axes[3].set_xlabel('Epsilon')
axes[3].set_ylabel('Noise Standard Deviation')
axes[3].set_title('Noise Magnitude vs Privacy Budget')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# Plot 5: MAE vs Epsilon
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[4].plot(df_results[mask]['epsilon'], df_results[mask]['mae'], 
                    marker='v', label=mechanism.capitalize())
axes[4].set_xscale('log')
axes[4].set_xlabel('Epsilon')
axes[4].set_ylabel('Mean Absolute Error')
axes[4].set_title('MAE vs Privacy Budget')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

# Plot 6: RMSE vs Epsilon
for mechanism in privacy_mechanisms:
    if mechanism != 'none':
        mask = df_results['mechanism'] == mechanism
        axes[5].plot(df_results[mask]['epsilon'], df_results[mask]['rmse'], 
                    marker='*', label=mechanism.capitalize())
axes[5].set_xscale('log')
axes[5].set_xlabel('Epsilon')
axes[5].set_ylabel('Root Mean Squared Error')
axes[5].set_title('RMSE vs Privacy Budget')
axes[5].legend()
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print comprehensive results table
print("\n" + "="*100)
print("DATA-LEVEL DIFFERENTIAL PRIVACY COMPARISON")
print("="*100)
print(f"{'Mechanism':<10} {'ε':<6} {'MSE':<8} {'MAE':<6} {'RMSE':<6} {'SNR':<6} {'Noise STD':<8} {'Pearson':<8}")

for mechanism in privacy_mechanisms:
    for epsilon in epsilon_values:
        mask = (df_results['mechanism'] == mechanism) & (df_results['epsilon'] == epsilon)
        if mask.any():
            row = df_results[mask].iloc[0]
            print(f"{mechanism:<10} {epsilon:<6} {row['mse']:<8.3f} {row['mae']:<6.3f} "
                  f"{row['rmse']:<6.3f} {row['snr']:<6.3f} {row['noise_std']:<8.3f} {row['pearson']:<8.3f}")

# Show sample of original vs noisy data
print(f"\n{'='*50}")
print("SAMPLE DATA COMPARISON")
print(f"{'='*50}")

# Get a sample document
sample_doc_idx = 0
original_doc = X_sample[sample_doc_idx].toarray().flatten()
noisy_docs = {}

for mechanism in [m for m in privacy_mechanisms if m != 'none']:
    X_noisy = apply_dp_to_data(X_sample, mechanism, epsilon=1.0)
    noisy_docs[mechanism] = X_noisy[sample_doc_idx].flatten()

# Show top features
top_feature_indices = np.argsort(original_doc)[-10:][::-1]  # Top 10 features

print(f"\nTop features for document {sample_doc_idx}:")
print(f"{'Feature':<15} {'Original':<10} {'Laplace':<10} {'Gaussian':<10} {'Staircase':<10}")
for idx in top_feature_indices:
    if original_doc[idx] > 0:
        feature_name = vectorizer.get_feature_names_out()[idx]
        print(f"{feature_name:<15} {original_doc[idx]:<10.1f} "
              f"{noisy_docs['laplace'][idx]:<10.1f} {noisy_docs['gaussian'][idx]:<10.1f} "
              f"{noisy_docs['staircase'][idx]:<10.1f}")