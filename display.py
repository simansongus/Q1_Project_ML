import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('AverageResults.csv')

def plot_attribute_selection_metrics(data):
    data.set_index('Attribute Eval', inplace=True)

    attr_metrics = data[['Attribute Avg ROC Area', 'Attribute Avg TP', 'Attribute Avg Accuracy']]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.2
    x = np.arange(len(attr_metrics.index))

    ax1.bar(x - bar_width, attr_metrics['Attribute Avg ROC Area'], width=bar_width, label='ROC Area', color='blue', alpha=0.6)
    ax1.set_ylabel('ROC Area', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(x, attr_metrics['Attribute Avg TP'], width=bar_width, label='TP Rate', color='green', alpha=0.6)
    ax2.set_ylabel('TP Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax3 = ax1.twinx()
    ax3.spines['top'].set_position(('outward', 60))
    ax3.bar(x + bar_width, attr_metrics['Attribute Avg Accuracy'], width=bar_width, label='Accuracy', color='orange', alpha=0.6)
    ax3.set_ylabel('Accuracy', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    ax1.set_xlabel('Attribute Selection Algorithms')
    ax1.set_title('Average Metrics by Attribute Selection Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attr_metrics.index)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_classifier_metrics(data):
    classifier_metrics = data[['Classifier Alg', 'Classifier Avg ROC Area', 'Classifier Avg TP', 'Classifier Avg Accuracy']]
    
    classifier_metrics.set_index('Classifier Alg', inplace=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.2
    x = np.arange(len(classifier_metrics.index))

    ax1.bar(x - bar_width, classifier_metrics['Classifier Avg ROC Area'], width=bar_width, label='ROC Area', color='blue', alpha=0.6)
    ax1.set_ylabel('ROC Area', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(x, classifier_metrics['Classifier Avg TP'], width=bar_width, label='TP Rate', color='green', alpha=0.6)
    ax2.set_ylabel('TP Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax3 = ax1.twinx()
    ax3.spines['top'].set_position(('outward', 60))
    ax3.bar(x + bar_width, classifier_metrics['Classifier Avg Accuracy'], width=bar_width, label='Accuracy', color='orange', alpha=0.6)
    ax3.set_ylabel('Accuracy', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    ax1.set_xlabel('Classifier Models')
    ax1.set_title('Average Metrics by Classifier Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classifier_metrics.index)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.tight_layout()
    plt.show()

plot_attribute_selection_metrics(data)
plot_classifier_metrics(data)