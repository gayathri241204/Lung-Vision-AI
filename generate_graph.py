import matplotlib.pyplot as plt
import numpy as np

# Ungaloda unmaiyaana recalculated datas mattume inge payanpaduthappattullana
models = ['DenseNet121', 'ResNet50', 'EfficientNet-B0']
accuracy = [92.4131, 93.9410, 94.2571]       #[cite: 26]
precision = [92.9735, 94.2918, 94.6450]      #[cite: 26]
recall = [93.8105, 94.7419, 95.1635]         #[cite: 26]
f1_score = [93.3687, 94.4691, 94.8758]       #[cite: 26]
parameters = [6.9590, 23.5183, 4.0140]       # Million[cite: 26]
latency = [216.1880, 251.9800, 75.6803]      # ms/image
throughput = [4.6256, 3.9686, 13.2135]       # Images/sec

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#ffffff')

# ----------------- SUBPLOT 1: CLASSIFICATION PERFORMANCE METRICS -----------------
x = np.arange(len(models))
width = 0.18

rects1 = ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3b82f6')
rects2 = ax1.bar(x - 0.5*width, precision, width, label='Macro Precision', color='#0ea5e9')
rects3 = ax1.bar(x + 0.5*width, recall, width, label='Macro Recall', color='#f59e0b')
rects4 = ax1.bar(x + 1.5*width, f1_score, width, label='Macro F1', color='#10b981')

ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold', color='#1e293b')
ax1.set_title('Empirical Classification Performance Across Models', fontsize=12, fontweight='bold', pad=12, color='#0f172a')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold', fontsize=10.5)
ax1.set_ylim(88, 98)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', frameon=True)

# Bar labels
for rect in rects1 + rects2 + rects3 + rects4:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, rotation=45)

# ----------------- SUBPLOT 2: LATENCY VS ACCURACY (PARETO FRONTIER) -----------------
colors = ['#f59e0b', '#ef4444', '#10b981']

for i, model in enumerate(models):
    ax2.scatter(latency[i], accuracy[i], s=parameters[i]*45, color=colors[i], alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.annotate(f"{model}\n({latency[i]:.1f} ms, {accuracy[i]:.2f}%)\nParams: {parameters[i]:.2f}M", 
                 xy=(latency[i], accuracy[i]), 
                 xytext=(10, -5 if i!=2 else 10), 
                 textcoords="offset points",
                 fontweight='bold', fontsize=9, color='#0f172a')

# Champion highlight
ax2.axvline(x=100, color='#10b981', linestyle=':', label='Real-time Threshold (<100ms)')
ax2.set_xlabel('Inference Latency (ms/image) - Lower is Better', fontsize=11, fontweight='bold', color='#1e293b')
ax2.set_ylabel('Overall Accuracy (%) - Higher is Better', fontsize=11, fontweight='bold', color='#1e293b')
ax2.set_title('Latency vs Accuracy Trade-off (Bubble Size = Parameters)', fontsize=12, fontweight='bold', pad=12, color='#0f172a')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlim(40, 300)
ax2.set_ylim(91, 96)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig('static/images/model_benchmark_graph.png', dpi=300)
print("Graph generated successfully as static/images/model_benchmark_graph.png!")