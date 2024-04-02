import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.filters import sobel
from skimage.metrics import mean_squared_error


def calculate_centralities(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    pr = nx.pagerank(G, alpha=0.9)
    ec = nx.eigenvector_centrality_numpy(G, max_iter=100)
    bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    return np.mean(list(pr.values())), np.mean(list(ec.values())), np.mean(list(bc.values()))
    
def calculate_gmsd(true_image, generated_image):
    true_edges = sobel(true_image)
    generated_edges = sobel(generated_image)
    mse = mean_squared_error(true_edges, generated_edges)
    return np.sqrt(mse)

def comprehensive_eval(true_hr_matrices, predicted_hr_matrices):
    metrics = {
        'MAE': [],
        'PCC': [],
        'JSD': [],
        'PC': [],
        'EC': [],
        'BC': [],
        'PSNR': [],
        'GMSD': []
    }
    # Annotations about what each metric signifies
    annotations = {
        'MAE': 'Lower is better. Measures average magnitude of errors between pairs of observations.',
        'PCC': 'Range [-1, 1]. Measures linear correlation between predictions and targets.',
        'JSD': 'Measures similarity between two probability distributions. Closer to 0 is better.',
        'PC': 'Measures PageRank centrality. Higher values indicate more central nodes.',
        'EC': 'Measures eigenvector centrality. Higher values indicate influential nodes in the network.',
        'BC': 'Measures betweenness centrality. Higher values indicate nodes that serve as bridges.',
        'GMSD': 'Lower is better. Measures the gradient magnitude similarity deviation, indicating edge preservation.'
    }
    
    for i in range(len(true_hr_matrices)):
        true_matrix = true_hr_matrices[i]
        pred_matrix = predicted_hr_matrices[i]

        # Flatten matrices for certain metrics
        true_flat = true_matrix.flatten()
        pred_flat = pred_matrix.flatten()

        # Calculate traditional and graph-based metrics
        metrics['MAE'].append(mean_absolute_error(true_flat, pred_flat))
        metrics['PCC'].append(pearsonr(true_flat, pred_flat)[0] if len(true_flat) > 0 and len(pred_flat) > 0 else np.nan)
        metrics['JSD'].append(jensenshannon(true_flat, pred_flat))
        
        try:
            pred_pc, pred_ec, pred_bc = calculate_centralities(pred_matrix)
            metrics['PC'].append(pred_pc)
            metrics['EC'].append(pred_ec)
            metrics['BC'].append(pred_bc)
        except Exception as e:
            print(f"Error calculating centralities for matrix {i}: {e}")
            # Append np.nan or some default value if error occurs
            metrics['PC'].append(np.nan)
            metrics['EC'].append(np.nan)
            metrics['BC'].append(np.nan)
        
        metrics['GMSD'].append(calculate_gmsd(true_matrix, pred_matrix))
    
    # Convert lists to means
    # Use np.nanmean to ignore NaN values and avoid mean of empty slice warning
    for key in metrics.keys():
        # Check if the list for the current metric contains any non-NaN values
        if np.isnan(metrics[key]).all():
            # Setting the metric to np.nan or to a specific value like 0
            metrics[key] = np.nan  
        else:
            metrics[key] = np.nanmean(metrics[key])
       
    return metrics

def plot_fold_metrics(fold_results):
    # Compute the average and standard deviation of the metrics across folds
    avg_metrics = {metric: np.mean([result[metric] for result in fold_results]) for metric in fold_results[0]}
    std_metrics = {metric: np.std([result[metric] for result in fold_results]) for metric in fold_results[0]}
    
    # Make sure you have as many colors as you have metrics
    num_metrics = len(fold_results[0])
    colors = sns.color_palette("hsv", num_metrics)
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten the axis array for easy iteration
    axs = axs.ravel()
    
    # Iterate over the folds and plot each one in a subplot
    for i, metrics in enumerate(fold_results):
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        # Plot each metric with a different color
        for j, key in enumerate(keys):
            axs[i].bar(key, values[j], color=colors[j])
        
        axs[i].set_title(f'Fold {i+1}')
        axs[i].xaxis.set_major_locator(plt.FixedLocator(range(num_metrics)))
        axs[i].xaxis.set_major_formatter(plt.FixedFormatter(keys))
        axs[i].tick_params(axis='x', rotation=45)
    
    # Plot the average metrics with error bars in the last subplot, using the same colors
    for i, (key, value) in enumerate(avg_metrics.items()):
        axs[-1].bar(key, value, yerr=std_metrics[key], color=colors[i], capsize=5)
    
    axs[-1].set_title('Avg. Across Folds')
    axs[-1].xaxis.set_major_locator(plt.FixedLocator(range(num_metrics)))
    axs[-1].xaxis.set_major_formatter(plt.FixedFormatter(list(avg_metrics.keys())))
    axs[-1].tick_params(axis='x', rotation=45)
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Show plot
    plt.show()
