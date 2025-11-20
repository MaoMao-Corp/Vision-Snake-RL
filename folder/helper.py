import matplotlib.pyplot as plt
from IPython import display
import os
import datetime

plt.ion()

def update_ema(previous_ema, new_value, alpha=0.1):
    if previous_ema is None:
        return new_value
    return alpha * new_value + (1 - alpha) * previous_ema

def plot(scores, ema_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(ema_scores, label='EMA Score')
    plt.ylim(ymin=0)

    # annotate latest values
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(ema_scores)-1, ema_scores[-1], f"{ema_scores[-1]:.2f}")

    plt.legend()
    plt.show(block=False)
    plt.pause(.1)

def save_final_plot(scores, ema_scores, name):
    # Create plots directory if it doesn't exist
    plots_folder_path = './plots'
    if not os.path.exists(plots_folder_path):
        os.makedirs(plots_folder_path)
    
    # Create figure for saving
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Score', alpha=0.7)
    plt.plot(ema_scores, label='EMA Score', linewidth=2)
    plt.title(f'{name} - Final Results (Games: {len(scores)})')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    if scores:
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        plt.text(0.02, 0.98, f'Max Score: {max_score}\nAvg Score: {avg_score:.2f}\nTotal Games: {len(scores)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_final_{timestamp}_{name}.png"
    filepath = os.path.join(plots_folder_path, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final plot saved as: {filename}")