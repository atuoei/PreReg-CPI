import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Global font settings
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14      # Base font size

def plot_volcano(enrich_df,
                 or_col="OR", q_col="q", name_col="alert",
                 or_thr=1.5, q_thr=0.05,
                 title="Enrichment Volcano Plot",
                 annotate_top_k=10, save_path=None,
                 figsize=(11, 8), dpi=300):
    """
    Enhanced volcano plot function (Arial font)
    """
    # Global font and font size
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14
    
    df = enrich_df.copy()
    eps = 1e-300
    
    # Compute transformed values
    df["log2OR"] = np.log2(df[or_col].astype(float) + eps)
    df["neglog10q"] = -np.log10(df[q_col].astype(float) + eps)
    
    # Determine significance
    df["sig"] = (df[or_col] >= or_thr) & (df[q_col] <= q_thr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Non-significant points
    non_sig = df[~df["sig"]]
    if len(non_sig) > 0:
        ax.scatter(non_sig["log2OR"], non_sig["neglog10q"], 
                  s=180, alpha=0.6, color='#6c757d', 
                  edgecolors='white', linewidth=0.5, label="Non-significant")
    
    # Significant points
    sig_df = df[df["sig"]]
    if len(sig_df) > 0:
        colors = plt.cm.Reds_r(np.linspace(0.1, 0.7, len(sig_df)))
        sizes = np.linspace(250, 180, len(sig_df))
        
        for i, (_, row) in enumerate(sig_df.iterrows()):
            ax.scatter(row["log2OR"], row["neglog10q"], 
                      s=sizes[i], alpha=0.85, color=colors[i],
                      edgecolors='white', linewidth=0.8, zorder=5)
    
    # Threshold lines
    ax.axvline(np.log2(or_thr), linestyle="--", linewidth=1, color='#dc3545', alpha=0.7)
    ax.axhline(-np.log10(q_thr), linestyle="--", linewidth=1, color='#dc3545', alpha=0.7)
    
    # Axis labels
    ax.set_xlabel("log2(Odds Ratio)", fontsize=16, fontweight='bold')
    ax.set_ylabel("-log10(FDR q-value)", fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.2)
    
    # Statistics text
    total_points = len(df)
    sig_points = len(sig_df)
    sig_percent = (sig_points / total_points) * 100 if total_points > 0 else 0
    stats_text = f"Total: {total_points}\nSignificant: {sig_points} ({sig_percent:.1f}%)"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Legend
    if len(sig_df) > 0:
        legend_elements = [
            mpatches.Patch(color='#6c757d', alpha=0.6, label='Non-significant'),
            mpatches.Patch(color='#dc3545', alpha=0.8, label='Significant')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                 fancybox=True, shadow=True, fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor='white')
    plt.show()
    return fig, ax


import pandas as pd

if __name__ == "__main__":
    # Create example data and plot
    example_df = pd.read_csv('enrich_broad.csv')
    
    # Generate enhanced volcano plot
    plot_volcano(example_df, or_col='OR', q_col='q', name_col='alert',
                title="Volcano Plot",
                annotate_top_k=9, save_path="Volcano Plot.svg")
