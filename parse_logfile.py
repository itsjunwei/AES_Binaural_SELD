import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(log_file_path):
    """
    Parses the log file and extracts epoch-wise metrics.

    Parameters:
        log_file_path (str): Path to the log .txt file.

    Returns:
        pd.DataFrame: DataFrame containing the extracted metrics.
    """
    # Initialize lists to store metrics
    epochs = []
    val_losses = []
    f_scores = []
    a_errors = []
    d_errors = []
    rde_errors = []
    seld_errors = []

    # Regular expression to match epoch lines
    epoch_pattern = re.compile(
        r"epoch:\s*(\d+),.*?val_loss:\s*([0-9.]+),.*?F/AE/DE/RDE/SELD:\s*([0-9.nan]+)/([0-9.nan]+)/([0-9.nan]+)/([0-9.nan]+)/([0-9.nan]+)"
    )

    with open(log_file_path, 'r') as file:
        for line in file:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                f_score = match.group(3)
                ae = match.group(4)
                de = match.group(5)
                rde = match.group(6)
                seld = match.group(7)

                # Convert 'nan' strings to actual NaN
                f_score = float(f_score) if f_score.lower() != 'nan' else 0
                ae = float(ae) if ae.lower() != 'nan' else 0
                de = float(de) if de.lower() != 'nan' else 0
                rde = float(rde) if rde.lower() != 'nan' else 0
                seld = float(seld) if seld.lower() != 'nan' else 0

                # Append to lists
                if epoch >= 1:
                    epochs.append(epoch)
                    val_losses.append(val_loss)
                    f_scores.append(f_score)
                    a_errors.append(ae)
                    d_errors.append(de)
                    rde_errors.append(rde)
                    seld_errors.append(seld)

    # Create DataFrame
    data = {
        'Epoch': epochs,
        'Val_Loss': val_losses,
        'F_score': f_scores,
        'AE': a_errors,
        'DE': d_errors,
        'RDE': rde_errors,
        'SELD': seld_errors
    }

    df = pd.DataFrame(data)
    return df

def compute_moving_average(df, window_size=5):
    """
    Computes the moving average for each metric.

    Parameters:
        df (pd.DataFrame): DataFrame containing the metrics.
        window_size (int): Window size for the moving average.

    Returns:
        pd.DataFrame: DataFrame with additional smoothed columns.
    """
    smoothed_df = df.copy()
    metrics = ['Val_Loss', 'F_score', 'AE', 'DE', 'RDE', 'SELD']
    for metric in metrics:
        smoothed_metric = f"{metric}_Smoothed"
        smoothed_df[smoothed_metric] = smoothed_df[metric].rolling(window=window_size, min_periods=1).mean()
    return smoothed_df

def plot_metrics(df, window_size=5, save_fig=False, output_path='training_metrics.png'):
    """
    Plots the metrics with smoothed lines.

    Parameters:
        df (pd.DataFrame): DataFrame containing the metrics and smoothed metrics.
        window_size (int): Window size used for smoothing.
        save_fig (bool): Whether to save the figure as an image file.
        output_path (str): Path to save the figure if save_fig is True.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 25))

    metrics = [
        ('Val_Loss', 'Val_Loss_Smoothed', 'Validation Loss', 'blue'),
        ('F_score', 'F_score_Smoothed', 'F-score', 'green'),
        ('AE', 'AE_Smoothed', 'Angular Error (AE)', 'red'),
        ('DE', 'DE_Smoothed', 'Distance Error (DE)', 'purple'),
        ('RDE', 'RDE_Smoothed', 'Relative Distance Error (RDE)', 'orange'),
        ('SELD', 'SELD_Smoothed', 'SELD Error', 'brown')
    ]

    for idx, (original, smoothed, title, color) in enumerate(metrics, 1):
        plt.subplot(3, 2, idx)
        sns.lineplot(x='Epoch', y=original, data=df, label='Original', color=color, marker='o')
        sns.lineplot(x='Epoch', y=smoothed, data=df, label=f'Smoothed (Window={window_size})', color='black', linestyle='--')
        plt.title(f'{title} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path, dpi=300)
        print(f"Plots saved to {output_path}")
    plt.show()

def main():
    """
    Main function to execute the script.
    """
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py path_to_log_file.txt")
        sys.exit(1)

    import os
    log_file_path = os.path.join("logs", sys.argv[1])
    if not log_file_path.endswith(".txt"):
        log_file_path += ".txt"

    # Parse the log file
    print(f"Parsing log file: {log_file_path}")
    df = parse_log_file(log_file_path)
    print("Parsing completed. Extracted data:")
    print(df.head())

    # Compute moving averages
    window_size = 5  # You can modify the window size as needed
    df_smoothed = compute_moving_average(df, window_size=window_size)
    print("Computed moving averages.")

    # Plot the metrics
    save_fig = False  # Set to True to save the plots
    output_path = 'training_metrics.png'  # Specify the output path if saving
    plot_metrics(df_smoothed, window_size=window_size, save_fig=save_fig, output_path=output_path)

if __name__ == "__main__":
    main()
