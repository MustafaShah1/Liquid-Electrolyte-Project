import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(data, window_size=3):
    # Simple moving average smoothing
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Simulation info
simulations = [
    {"data": "system.data", "traj": "unwrapped.dcd", "type_a": 13, "type_b": 14, "label": "1.5M LiTFSI"}
]

rdf_results = []
coordination_results = []

for sim in simulations:
    u = mda.Universe(sim["data"], sim["traj"])
    group_a = u.select_atoms(f"type {sim['type_a']}")
    group_b = u.select_atoms(f"type {sim['type_b']}")

    n_frames = len(u.trajectory)
    # Use up to two evenly spaced frames, but handle short trajectories gracefully
    if n_frames < 2:
        selected_frames = [0]
    else:
        selected_frames = np.linspace(0, n_frames - 1, 2, dtype=int)
   
    rdf_list = []
    volumes = [u.trajectory[frame].volume for frame in selected_frames]
    mean_volume = np.mean(volumes)
    number_density_b = len(group_b) / mean_volume

    bins = None  # Will set after first RDF calculation

    for frame in selected_frames:
        u.trajectory[frame]
        rdf = InterRDF(group_a, group_b, nbins=75, range=(0.0, 15.0))
        rdf.run()
        rdf_list.append(rdf.results.rdf.copy())
        if bins is None:
            bins = rdf.results.bins.copy()

    rdf_array = np.array(rdf_list)
    mean_rdf = np.mean(rdf_array, axis=0)
    mean_rdf = smooth_curve(mean_rdf, window_size=3) # smoothing
   
    dr = bins[1] - bins[0]
    coordination_number = np.cumsum(mean_rdf * 4 * np.pi * bins**2 * dr) * number_density_b
    coordination_number = smooth_curve(coordination_number, window_size=5) # smoothing

    rdf_results.append((sim["label"], bins, mean_rdf))
    coordination_results.append((sim["label"], bins, coordination_number))

fig, ax1 = plt.subplots(figsize=(12, 8))

# Set thick axis spines on primary axis
for spine in ax1.spines.values():
    spine.set_linewidth(3)

colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors

# Plot RDF curves on primary y-axis with styled labels
for i, (label, bins, mean_rdf) in enumerate(rdf_results):
    color = colors[i % len(colors)]
    ax1.plot(bins, mean_rdf, color=color, linestyle='-', linewidth=3, label=f'g(r) {label}')

ax1.set_xlabel('r/(Å)', fontsize=24, fontweight='bold', labelpad=15)
ax1.set_ylabel(r'$\mathbf{g_{Li^{+}-O(Anion)}(r)}$', fontsize=28, fontweight='bold', labelpad=15)
ax1.tick_params(axis='both', which='major', labelsize=24, width=3, length=12)
ax1.tick_params(axis='both', which='minor', labelsize=24, width=3, length=8)
# Make all major and minor tick labels bold on both axes
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
#ax1.set_title('Mean RDF and Coordination Number Comparison', fontsize=24, fontweight='bold', pad=20)

# Create secondary y-axis for coordination number
ax2 = ax1.twinx()
# Plot coordination numbers on secondary y-axis
for i, (label, bins, coordination_number) in enumerate(coordination_results):
    color = colors[i % len(colors)]
    ax2.plot(bins, coordination_number, color=color, linestyle='--', linewidth=3, label=f'N(r) {label}')
ax2.set_ylabel(r'$\mathbf{N_{Li^{+}-O(Anion)}(r)}$', fontsize=28, fontweight='bold', labelpad=15)
ax2.tick_params(axis='y', labelsize=24, width=3, length=12)
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

# Combine legends and style
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=20)
for text in legend.get_texts():
    text.set_fontweight('bold')

ax1.set_xlim(0, 14)  
ax2.set_ylim(0, 12)        
plt.tight_layout(pad=3.0)
plt.show()