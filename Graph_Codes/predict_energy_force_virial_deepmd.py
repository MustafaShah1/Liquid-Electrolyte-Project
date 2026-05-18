import matplotlib
matplotlib.use('Agg')  

import dpdata
import plotly.graph_objects as go
import numpy as np
import pickle
from plotly.subplots import make_subplots
import glob
import matplotlib.pyplot as plt
import re
import random
import os

# Load the training systems
training_systems = dpdata.LabeledSystem("../00data/validation_data", fmt="deepmd/npy")

### -------------------Energy------------------- ###

# Check if the prediction file exists
if os.path.exists('predict.pkl'):
    with open('predict.pkl', 'rb') as f:
        predict = pickle.load(f)
else:
    predict = training_systems.predict("graph-compress.pb")
    with open('predict.pkl', 'wb') as f:
        pickle.dump(predict, f)

# Extract energies
energies_aimd = training_systems["energies"]
energies_predicted = predict["energies"]

# Create scatter plot
scatter = go.Scatter(x=energies_aimd, y=energies_predicted, mode='markers', name='AIMD vs Predicted',
                     marker=dict(size=10))  # Adjust marker size for visibility

# Create reference line
x_range = np.linspace(min(energies_aimd), max(energies_aimd), 100)
line = go.Scatter(x=x_range, y=x_range, mode='lines', name='Reference Line',
                  line=dict(color='red', dash='dash'))

# Calculate R² value
y_mean = np.mean(energies_aimd)
SST = np.sum((energies_aimd - y_mean)**2)
SSE = np.sum((energies_aimd - energies_predicted)**2)
R_squared = 1 - (SSE / SST)

# Define layout
layout = go.Layout(
    title=dict(
        text='Energy Comparison: AIMD vs Deep Potential Prediction',
        font=dict(size=24, family='Helvetica Bold'),  # Set title to be larger and bold
        x=0.5  # Center title
    ),
    xaxis=dict(
        title='Energy of AIMD',
        tickfont=dict(size=20, family='Helvetica Bold'),  # Set tick labels to be smaller
        titlefont=dict(size=24, family='Helvetica Bold')  # Set title font size smaller
    ),
    yaxis=dict(
        title='Energy Predicted by Deep Potential',
        tickfont=dict(size=20, family='Helvetica Bold'),  # Set tick labels to be smaller
        titlefont=dict(size=24, family='Helvetica Bold')  # Set title font size smaller
    ),
    width=800,  # Set the width of the plot
    height=600,  # Set the height of the plot
    showlegend=True,
    legend=dict(x=1, y=1, bordercolor="Black", borderwidth=2)
)

# Create figure
fig = go.Figure(data=[scatter, line], layout=layout)

# Add R² annotation
fig.add_annotation(x=1, y=1, xref="paper", yref="paper",
                   text=f'R² = {R_squared:.6f}',
                   showarrow=False,
                   xanchor='right', yanchor='top',
                   font=dict(size=16, color="black"),
                   bgcolor="white",
                   bordercolor="black",
                   borderwidth=2)

# Save as HTML
fig.write_html('deepmd_energy_comparison.html')

# Save as PNG
fig.write_image('deepmd_energy_comparison.png', width=800, height=600, scale=2, engine='kaleido')



### -------------------Force------------------- ###

number_of_frames_to_consider = 1000

forces_aimd = training_systems["forces"]
if os.path.exists('predict.pkl'):
    with open('predict.pkl', 'rb') as f:
        predict = pickle.load(f)
else:
    predict = training_systems.predict("graph.pb")
    with open('predict.pkl', 'wb') as f:
        pickle.dump(predict, f)
forces_ml = predict["forces"]

with open("training_aimd.txt", "w") as f:
    for item in training_systems:
        f.write("%s\n" % item)
with open("training_ml.txt", "w") as f:
    for item in predict:
        f.write("%s\n" % item)

def save_array_to_file(array, filename):
    with open(filename, "w") as f:
        for item in array:
            str_item = np.array2string(item, separator='  ', threshold=np.inf, max_line_width=np.inf).replace('[', '').replace(']', '')
            f.write(str_item + "\n")

save_array_to_file(forces_aimd, "forces_aimd_all.txt")
save_array_to_file(forces_ml, "forces_ml_all.txt")


def extract_atom_counts(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 8:
                elements = line.strip().split()
            elif i == 9:
                counts = [int(x) for x in line.strip().split()]
                break
    return dict(zip(elements, counts))

def categorize_forces(input_file, atom_counts, output_prefix):
    total_atoms = sum(atom_counts.values())
    with open(input_file, 'r') as file:
        lines = file.readlines()

    file_pointers = {element: open(f"{output_prefix}_{element}.txt", 'w') for element in atom_counts}

    start_idx = 0
    while start_idx < len(lines):
        for element, count in atom_counts.items():
            element_lines = lines[start_idx:start_idx+count]
            file_pointers[element].writelines(element_lines)
            start_idx += count

    for fp in file_pointers.values():
        fp.close()

atom_counts_aimd = extract_atom_counts("./training_aimd.txt")
atom_counts_ml = extract_atom_counts("./training_ml.txt")
total_atoms_aimd = sum(atom_counts_aimd.values())
total_atoms_ml = sum(atom_counts_ml.values())

def random_select_and_write_parts(file_a_path, file_e_path, c, d, file_b_path, file_f_path):
    # 读取文件A和E的内容
    with open(file_a_path, 'r') as file_a, open(file_e_path, 'r') as file_e:
        lines_a = file_a.readlines()
        lines_e = file_e.readlines()

    # 确保文件A和E的行数相同
    assert len(lines_a) == len(lines_e), "文件A和E的行数不相同！"

    # 划分为部分
    parts = len(lines_a) // c
    indices = list(range(parts))
    
    # 随机选取D部分
    selected_indices = random.sample(indices, min(d, parts))

    # 写入新文件B和F
    with open(file_b_path, 'w') as file_b, open(file_f_path, 'w') as file_f:
        for index in selected_indices:
            start_line = index * c
            end_line = start_line + c
            file_b.writelines(lines_a[start_line:end_line])
            file_f.writelines(lines_e[start_line:end_line])

# 示例调用
file_a_path = './forces_aimd_all.txt'  # 替换为文件A的路径
file_e_path = './forces_ml_all.txt'  # 替换为文件E的路径
c = total_atoms_aimd  # 每C行为一部分
d = number_of_frames_to_consider  # 随机选取D部分
file_b_path = './forces_aimd.txt'  # 替换为要创建的文件B的路径
file_f_path = './forces_ml.txt'  # 替换为要创建的文件F的路径

random_select_and_write_parts(file_a_path, file_e_path, c, d, file_b_path, file_f_path)

categorize_forces_aimd=categorize_forces("./forces_aimd.txt", atom_counts_aimd, "./forces")
categorize_forces_ml=categorize_forces("./forces_ml.txt", atom_counts_ml, "./forces")

with open("./forces_aimd.txt", "r") as file:
    lines = file.readlines()

if len(lines) % total_atoms_aimd != 0:
    print("Warning: The total number of lines in forces_aimd.txt is not a multiple of the total atom count. Check the data.")
else:
    for cycle in range(len(lines) // total_atoms_aimd):
        start_line = cycle * total_atoms_aimd
        for element, count in atom_counts_aimd.items():
            end_line = start_line + count
            element_lines = lines[start_line:end_line]
            start_line = end_line
            with open(f"./forces_aimd_{element}_{cycle}.txt", "w") as out_file:
                out_file.writelines(element_lines)


with open("./forces_ml.txt", "r") as file:
    lines = file.readlines()

if len(lines) % total_atoms_ml != 0:
    print("Warning: The total number of lines in forces_ml.txt is not a multiple of the total atom count. Check the data.")
else:
    for cycle in range(len(lines) // total_atoms_ml):
        start_line = cycle * total_atoms_ml
        for element, count in atom_counts_ml.items():
            end_line = start_line + count
            element_lines = lines[start_line:end_line]
            start_line = end_line
            with open(f"./forces_ml_{element}_{cycle}.txt", "w") as out_file:
                out_file.writelines(element_lines)

def clean_line(line):
    return re.sub(r'[\[\]]', '', line)

def read_forces(pattern):
    forces = {}
    for file in glob.glob(pattern):
        parts = file.split('_')
        element, cycle = parts[2], parts[3].split('.')[0]
        key = (element, cycle)
        with open(file, 'r') as f:
            data = np.array([list(map(float, clean_line(line).split())) for line in f if line.strip()])
            forces[key] = data
    return forces

def calculate_R_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    SST = np.sum((y_true - y_mean)**2)
    SSE = np.sum((y_true - y_pred)**2)
    R_squared = 1 - (SSE / SST)
    return R_squared

def plot_all_elements_forces(forces_aimd, forces_ml, downsample=True):
    elements = sorted(set(key[0] for key in forces_aimd.keys()))
    directions = ['X', 'Y', 'Z']
    n_elements = len(elements)

    # Prepare for matplotlib plot
    fig_matplotlib, axs = plt.subplots(n_elements, len(directions), figsize=(15, 5*n_elements), squeeze=False)
    colors_matplotlib = ['blue', 'orange', 'green']

    # Prepare for plotly plot
    fig_plotly = make_subplots(rows=n_elements, cols=len(directions), subplot_titles=[f"{el} {dir}" for el in elements for dir in directions])

    for i, el in enumerate(elements):
        for j, dir in enumerate(directions):
            row, col = i + 1, j + 1
            aimd_forces = np.array([])
            ml_forces = np.array([])
            for key in forces_aimd.keys():
                if key[0] == el:
                    aimd_forces = np.concatenate((aimd_forces, forces_aimd[key][:, j])) if aimd_forces.size else forces_aimd[key][:, j]
                    if key in forces_ml:
                        ml_forces = np.concatenate((ml_forces, forces_ml[key][:, j])) if ml_forces.size else forces_ml[key][:, j]

            if len(aimd_forces) > 0 and len(ml_forces) > 0:
                R_squared = calculate_R_squared(aimd_forces, ml_forces)

                # Plot with matplotlib
                axs[i, j].scatter(aimd_forces, ml_forces, color=colors_matplotlib[j], alpha=0.6)
                axs[i, j].set_title(f"{el} {dir} Direction, R² = {R_squared:.2f}", fontsize=20, fontweight='bold')
                axs[i, j].set_xlabel(f"DFT F_{dir} (eV/Å)", fontsize=20, fontweight='bold')
                axs[i, j].set_ylabel(f"ML F_{dir} (eV/Å)", fontsize=20, fontweight='bold')
                axs[i, j].tick_params(axis='both', which='major', labelsize=15)

                # Plot with plotly
                trace = go.Scatter(x=aimd_forces, y=ml_forces, mode='markers', name=f'{el} {dir}',
                                   marker=dict(color=colors_matplotlib[j]))
                fig_plotly.add_trace(trace, row=row, col=col)
                fig_plotly.layout.annotations[(row-1) * len(directions) + (col-1)].update(text=f"{el} {dir} Direction, R² = {R_squared:.2f}")
                fig_plotly.update_xaxes(title_font=dict(size=30, family="Arial bold"))
                fig_plotly.update_yaxes(title_font=dict(size=30, family="Arial bold"))
                for annotation in fig_plotly.layout.annotations:
                    annotation.update(font=dict(size=40, family="Arial bold"))

    plt.tight_layout()
    plt.savefig('deepmd_force_comparison', dpi=300)
    plt.close(fig_matplotlib)

    fig_plotly.update_layout(height=300*n_elements, showlegend=False)
    fig_plotly.write_html('deepmd_force_comparison.html')

    return 'deepmd_force_comparison.png', 'deepmd_force_comparison.html'

# Set this to False if you want to use all data points
downsample = True

# Read the AIMD and ML model force data
forces_aimd = read_forces("./forces_aimd_*_*.txt")
forces_ml = read_forces("./forces_ml_*_*.txt")

# Plot the graphs
output_png, output_html = plot_all_elements_forces(forces_aimd, forces_ml, downsample)

# Delete txt files
for file_path in glob.glob("forces_aimd*"):
    os.remove(file_path)

for file_path in glob.glob("forces_ml*"):
    os.remove(file_path)

print(f"Plot saved to {output_png} and {output_html}")
