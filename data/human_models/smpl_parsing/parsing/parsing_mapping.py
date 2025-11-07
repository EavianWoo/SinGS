import os
import json

import numpy as np

mapping_dir = os.path.dirname(os.path.dirname(__file__))

smpl_vert_segmentation_path = os.path.join(mapping_dir, 'smpl_vert_segmentation.json')
with open(smpl_vert_segmentation_path, "r") as f:
    region_vertex_map = json.load(f)

label_region_map_path = os.path.join(mapping_dir, 'label_region_map.json')
with open(label_region_map_path, "r") as f:
    label_region_map = json.load(f)

region_label_map_path = os.path.join(mapping_dir, 'region_label_map.json')
with open(region_label_map_path, "r") as f:
    region_label_map = json.load(f)


def get_vertex_label(NUM_VERT):
    # assert NUM_VERT 
    label_vertex_map = {}
    v_label = -1 * np.ones(NUM_VERT)
    for label, regions in label_region_map.items():
        label = int(label) ###
        label_vertex_map[label] = []
        for region in regions:
            label_vertex_map[label].extend(region_vertex_map[region])
            v_label[region_vertex_map[region]] = label

    return v_label

def parse_weights(weight_dict):
    weights = np.ones(len(weight_dict))
    
    for region, label in region_label_map.items():
        weights[label] = weight_dict[region]
    
    return weights
        