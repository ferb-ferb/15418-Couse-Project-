import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. DATA SETUP
# ---------------------------------------------------------
# Ordered by total time (fastest to slowest) to follow best practices
labels = ['Spatial Hash', 'Neighbor List', 'Brute Force']

# Timings in milliseconds (Reordered to match labels above)
build_times = np.array([0.02, 93.56, 0.0])
density_times = np.array([2.38, 0.30, 47.03])
forces_times = np.array([7.14, 1.69, 129.03])

# Calculate Totals
total_times = build_times + density_times + forces_times

# Calculate Speedups (Relative to Brute Force)
brute_density = 47.03
brute_forces = 129.03

speedup_density = brute_density / density_times
speedup_forces = brute_forces / forces_times

# ---------------------------------------------------------
# 2. PLOT: STACKED BAR CHART (TOTAL TIME BREAKDOWN)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

width = 0.6
# Create stacked bars
p1 = ax.bar(labels, build_times, width, label='Build Search Structure', color='#d62728', edgecolor='black')
p2 = ax.bar(labels, density_times, width, bottom=build_times, label='Compute Density/Pressure', color='#1f77b4', edgecolor='black')
p3 = ax.bar(labels, forces_times, width, bottom=build_times + density_times, label='Compute Forces', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('SPH Substep Execution Time Breakdown (56k Particles)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

# Add text labels on top of the bars with the total time
for i, total in enumerate(total_times):
    ax.text(i, total + 2, f'{total:.2f} ms', ha='center', fontweight='bold')

# Ensure layout is not truncated and save
plt.tight_layout()
plt.savefig('total_time_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. PLOT: GROUPED BAR CHART (KERNEL SPEEDUPS)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, speedup_density, width, label='Density Kernel Speedup', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, speedup_forces, width, label='Forces Kernel Speedup', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Speedup (Relative to Brute Force)', fontsize=12, fontweight='bold')
ax.set_title('Math Kernel Speedup by Implementation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)

# Add a horizontal line at 1.0 to show the baseline
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Brute Force Baseline (1.0x)')

# Add text labels for the speedup values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}x',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('kernel_speedups.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figures generated successfully: 'total_time_stacked.png' and 'kernel_speedups.png'")
