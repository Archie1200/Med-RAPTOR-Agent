import matplotlib.pyplot as plt
import numpy as np

# Sample accuracy data for three RAG approaches
# Replace these values with your actual experimental results
approaches = ['RAG', 'Raptor Agent', 'Raptor Agent + NLI verification']
accuracies = [20, 60, 80]  # Example accuracies in percentage

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar plot
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(approaches, accuracies, color=colors, alpha=1, edgecolor='black', linewidth=0.5, width=0.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the plot
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('RAG Approach', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Comparison of Different RAG Approaches', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Rotate x-axis labels if needed
plt.xticks(rotation=0, ha='right')

plt.tight_layout()
plt.savefig('rag_accuracy_bar_chart.png', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'rag_accuracy_bar_chart.png'")
plt.show()

# Optional: Create a line plot for trend visualization
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(approaches, accuracies, marker='o', linewidth=2.5, 
         markersize=10, color='#2c3e50', markerfacecolor='#e74c3c', 
         markeredgewidth=2, markeredgecolor='#2c3e50')

# Add value labels
for i, (approach, acc) in enumerate(zip(approaches, accuracies)):
    ax2.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('RAG Approach', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Improvement Across RAG Approaches', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('rag_accuracy_line_plot.png', dpi=300, bbox_inches='tight')
print("Line plot saved as 'rag_accuracy_line_plot.png'")
plt.show()

# Print improvement statistics
print("Accuracy Comparison:")
print("-" * 50)
for approach, acc in zip(approaches, accuracies):
    print(f"{approach:25s}: {acc:5.1f}%")
print("-" * 50)
print(f"Improvement (Raptor):     +{accuracies[1] - accuracies[0]:.1f}%")
print(f"Improvement (Raptor+NLI): +{accuracies[2] - accuracies[0]:.1f}%")
print(f"Relative improvement:     {((accuracies[2] - accuracies[0]) / accuracies[0] * 100):.1f}%")