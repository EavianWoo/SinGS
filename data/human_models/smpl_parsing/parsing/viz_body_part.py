import os
import sys
import json
import trimesh
import subprocess
import numpy as np
from smplx import SMPL, SMPLH, SMPLX
from matplotlib import cm as mpl_cm, colors as mpl_colors


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors


def main(body_model='smpl', body_model_path='./data/smplx_models/smpl/'):
    main_url = 'https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/'
    if body_model == 'smpl':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
        body_model = SMPL(model_path=body_model_path)
    elif body_model == 'smplx':
        part_segm_url = os.path.join(main_url, 'smplx/smplx_vert_segmentation.json')
        body_model = SMPLX(model_path=body_model_path)
    elif body_model == 'smplh':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
        body_model = SMPLH(model_path=body_model_path)
    else:
        raise ValueError(f'{body_model} is not defined, \"smpl\", \"smplh\" or \"smplx\" are valid body models')

    part_segm_filepath = download_url(part_segm_url, './viz/')
    part_segm = json.load(open(part_segm_filepath))


    vertices = body_model().vertices[0].detach().numpy()
    faces = body_model.faces

    vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])

    mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
    scene = trimesh.Scene(mesh)
    scene.show(background=(0, 0, 0, 0)) 


if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2])
    main()