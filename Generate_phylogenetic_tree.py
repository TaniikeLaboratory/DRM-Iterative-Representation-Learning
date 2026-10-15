import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree, TreeStyle, TextFace, NodeStyle
from PyQt5 import QtGui
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Inputs and settings
DATA_CSV = "./data/Catalyst_phylogenetic_tree_data.csv"
ELEMENT_DISTANCE_CSV = "./data/element_distance_matrix.csv"
alpha = 0.8
color = "H2 yield_max"
desired_dpi = 300
clip_value = 200
component_flag = "M"
highlight_elements = []
group_colors = []

os.makedirs("./figures", exist_ok=True)
tree_img = "./figures/phylogenetic_tree_base.png"
final_output_file = "./figures/phylogenetic_tree.png"

# Catalyst grouping and aggregation
df = pd.read_csv(DATA_CSV)
df = df.drop(columns="Note", errors="ignore")
def compute_name_and_groups(row):

    new_name=row[0]
    parts = new_name.split("/")
    m_elements = parts[0].split("-") if "-" in parts[0] else [parts[0]]
    s_elements = parts[1].split("-") if "-" in parts[1] else [parts[1]]

    return new_name, m_elements, s_elements

df[["Name", "M_group", "S_group"]] = df.apply(
    lambda row: pd.Series(compute_name_and_groups(row)), axis=1
)

df.set_index("Name", inplace=True)
df.index.name = None

if "Catalyst" in df.columns:
    df.drop(["Catalyst"], axis=1, inplace=True)

exclude_cols = ["M_group", "S_group"]
non_element_cols = [c for c in df.columns if c not in exclude_cols]

name_list = df.index.tolist()
name_counter = Counter(name_list)

df_for_agg = df[non_element_cols].copy()

df_for_agg["count"] = [name_counter[nm] for nm in df.index]

def mode(x):
    m = x.mode(dropna=True)
    return m.iloc[0] if len(m) > 0 else np.nan

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def total(x):
    return x.iloc[0]

aggregations = {}
for col in df_for_agg.columns:
    if (df_for_agg[col].name=="count"):
        aggregations[col] = [total]
    elif is_numeric_dtype(df_for_agg[col]):
        aggregations[col] = ['min','max','median','mean', mode, q25, q75]
    else:
        aggregations[col] = [mode]
df_agg = df_for_agg.groupby(level=0).agg(aggregations)

agg_name_map = {
    total: 'total',
    'max': 'max',
    'median': 'med',
    'mean': 'avg',
    mode: 'mode',
    q25: 'q25',
    q75: 'q75'
}

def get_agg_name(func):
    if func in agg_name_map:
        return agg_name_map[func]
    elif isinstance(func, str):
        return func
    else:
        return func.__name__

df_agg.columns = [
    f"{col[0]}_{get_agg_name(col[1])}"
    for col in df_agg.columns
]

valid_names = df_agg.index
df_for_distance = df.loc[valid_names, ["M_group", "S_group"]].copy()

df_for_distance = df_for_distance[~df_for_distance.index.duplicated(keep='first')]

M_groups = df_for_distance["M_group"].tolist()
S_groups = df_for_distance["S_group"].tolist()
Name_unique = df_for_distance.index.tolist()
Comp = list(zip(M_groups, S_groups))
N = len(Comp)

df_pt = pd.read_csv(ELEMENT_DISTANCE_CSV, index_col=0)
element_distance_matrix = df_pt

def compare(a, b):
    """Compute the symmetric distance between two element groups."""
    if not a and not b:
        return 0

    distance_a = element_distance_matrix.loc[a, b].min(axis=1).sum() if a else 0
    distance_b = element_distance_matrix.loc[b, a].min(axis=1).sum() if b else 0
    return distance_a + distance_b

def compute_distance_row(i):
    """Compute distances from catalyst i to catalysts j >= i."""
    distances = np.zeros(N)

    M_i, S_i = Comp[i]
    M_i_local = S_i if len(M_i) == 0 else M_i

    for j in range(i, N):
        M_j, S_j = Comp[j]
        M_j_local = S_j if len(M_j) == 0 else M_j

        d_m = compare(M_i_local, M_j_local)
        d_s = compare(S_i, S_j)
        total = alpha * d_m + (1 - alpha) * d_s
        distances[j] = total

    return i, distances

D = np.zeros((N, N))
results = Parallel(n_jobs=-1)(
    delayed(compute_distance_row)(i) for i in tqdm(range(N))
)

for i, distances in results:
    D[i, i:] = distances[i:]
    D[i:, i] = distances[i:]

if D.max() != D.min():
    D = (D - D.min()) / (D.max() - D.min())
D = np.round(D, 5)

df_distance = pd.DataFrame(D, index=Name_unique, columns=Name_unique)


# Select high-yield combinations for tree construction
df_stats = df_agg[df_agg["H2 yield_max"] >= 20]
remaining_catalysts = df_stats.index
df_distance = df_distance.loc[remaining_catalysts, remaining_catalysts]

# Tree visualization
if color not in df_stats.columns:
    raise ValueError(
        f"The specified column '{color}' does not exist in df_stats: {df_stats.columns.tolist()}"
    )

df_order = df_stats.sort_values(by=color)
key = df_stats[color]
label = r"H$_2$ yield (%)"

cmap_obj = plt.cm.YlGnBu

def color_dict(z, scale="linear", alpha=1, cmap_obj=None):
    if cmap_obj is None:
        cmap_obj = plt.cm.jet

    color_mapping = {}
    z_sorted = sorted(z)
    min_val, max_val = z_sorted[0], z_sorted[-1]
    range_val = max_val - min_val

    beta = alpha * 1.5
    if beta > 1:
        beta = 1

    alpha_int = int(alpha * 255)
    beta_int = int(beta * 255)

    if scale == "linear":
        for val in z_sorted:
            norm = 0.5 if range_val == 0 else (val - min_val) / range_val
            rgb = cmap_obj(norm)
            rgb_255 = [int(c * 255) for c in rgb[:3]]

            rgb_255_origin = rgb_255.copy()
            rgb_255_origin.append(beta_int)
            color_origin = QtGui.QColor(*tuple(rgb_255_origin))
            hex_color_origin = color_origin.name(QtGui.QColor.HexRgb)

            rgb_255.append(alpha_int)
            color_final = QtGui.QColor(*tuple(rgb_255))
            hex_color = color_final.name(QtGui.QColor.HexRgb)

            color_mapping[str(val)] = [hex_color, hex_color_origin]
    else:
        pass

    return color_mapping

unique_values = key.unique().astype(int)
min_val = int(unique_values.min())
max_val = int(unique_values.max())
if max_val > clip_value:
    max_val = clip_value
clipped_values = list(range(min_val, max_val + 1))

color_mapping = color_dict(clipped_values, scale="linear", alpha=0.4, cmap_obj=cmap_obj)

names = df_distance.index.tolist()
dm = DistanceMatrix(df_distance.values, names)
tree_newick = nj(dm, result_constructor=str)
t = Tree(tree_newick, format=1)

name_order = df_order.index.tolist()

for i in name_order:
    leaf_nodes = t.search_nodes(name=i)
    if not leaf_nodes:
        continue
    leaf_node = leaf_nodes[0]

    tk = 5
    special_nstyle = NodeStyle()
    try:
        m_part, s_part = i.split("/")
    except ValueError:
        m_part, s_part = i, i
    m_elems = m_part.split("-") if m_part else []
    s_elems = s_part.split("-") if s_part else []

    target_elems = m_elems if component_flag == "M" else s_elems
    found_color = None
    for group, c_ in zip(highlight_elements, group_colors):
        if any(elem in group for elem in target_elems):
            found_color = c_
            break
    if found_color is not None:
        special_nstyle["hz_line_color"] = found_color
        special_nstyle["vt_line_color"] = found_color
        tk = 7

    special_nstyle["hz_line_width"] = tk
    special_nstyle["vt_line_width"] = tk
    special_nstyle["size"] = 0
    current_node = leaf_node
    while current_node:
        current_node.set_style(special_nstyle)
        current_node = current_node.up
    val = key.loc[i]
    val_clipped = min(int(val), clip_value)
    midpoint = (max(clipped_values) + min(clipped_values)) / 2
    if df_stats.loc[i,"Flag_min"] ==-10:
        name_face = TextFace(i, fgcolor="grey", fsize=50, bold=True)
    elif df_stats.loc[i,"Flag_min"] >11:
        temp="[New] "+i
        if val_clipped > midpoint:
            name_face = TextFace(i, fgcolor="white", fsize=50, bold=True)
        else:
            name_face = TextFace(i, fgcolor="black", fsize=50, bold=True)
    else:
        if val_clipped > midpoint:
            name_face = TextFace(i, fgcolor="white", fsize=50, bold=True)
        else:
            name_face = TextFace(i, fgcolor="black", fsize=50, bold=True)
    color_values = color_mapping.get(str(val_clipped), None)

    if color_values is not None:
        name_face.background.color = color_values[0]

    leaf_node.add_face(name_face, column=1, position='aligned')

circular_style = TreeStyle()
circular_style.show_leaf_name = False
circular_style.mode = "c"
circular_style.draw_guiding_lines = True
circular_style.guiding_lines_type = 1
circular_style.guiding_lines_color = "Silver"
circular_style.show_scale = False

for node in t.traverse():
    if node.dist == 0 and not node.is_root():
        node.dist = 0.01

width_in_inches = 8
height_in_inches = 8
width_in_pixels = int(width_in_inches * desired_dpi)
height_in_pixels = int(height_in_inches * desired_dpi)

t.render(tree_img, w=width_in_pixels, h=height_in_pixels, units="px", tree_style=circular_style)

def create_colorbar(cmap_obj, values_range, filename, scale='linear',
                    orientation='vertical', height_in_pixels=1000):
    min_v = min(values_range)
    max_v = max(values_range)
    norm = plt.Normalize(vmin=min_v, vmax=max_v)

    fig_height = height_in_pixels / desired_dpi
    fig_width = 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.5, right=0.7, bottom=0.05, top=0.95)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
    cbar.set_label(label, rotation=270, labelpad=15)

    fig.savefig(filename, dpi=desired_dpi, bbox_inches='tight')
    plt.close(fig)

def combine_images(tree_image_filename, colorbar_image_filename, final_image_filename):
    tree_image = Image.open(tree_image_filename)
    colorbar_image = Image.open(colorbar_image_filename)

    tree_height = tree_image.height
    tree_width = tree_image.width
    colorbar_aspect_ratio = colorbar_image.width / colorbar_image.height
    new_colorbar_width = int(colorbar_aspect_ratio * tree_height)
    colorbar_image = colorbar_image.resize((new_colorbar_width, tree_height), Image.LANCZOS)

    total_width = tree_width + colorbar_image.width
    max_height = tree_height

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    new_im.paste(tree_image, (0, 0))
    new_im.paste(colorbar_image, (tree_width, 0))

    new_im.save(final_image_filename)

tree_image = Image.open(tree_img)
tree_height = tree_image.height

colorbar_img = "./figures/phylogenetic_tree_colorbar.png"
create_colorbar(cmap_obj, clipped_values, colorbar_img,
                orientation='vertical', height_in_pixels=tree_height)

combine_images(tree_img, colorbar_img, final_output_file)

# Remove temporary rendering images
os.remove(tree_img)
os.remove(colorbar_img)
