import torch 
import trimesh 
import numpy as np 
from trimesh import grouping 
from trimesh.geometry import faces_to_edges  


def subdivide_meshes(
    vertices,
    faces,              
    face_index=None,     
    # selected_v_label=[0, 1],          
    vertex_attributes=None
):     
    """Selective subdivision according to assigned face label."""
    device = faces.device
    if face_index is None:
        face_mask = torch.ones(len(faces), dtype=bool, device=device)     
    else:         
        face_mask = torch.zeros(len(faces), dtype=bool, device=device)         
        face_mask[face_index] = True 
    #     vertex_label = vertex_attributes['vertex_label'].squeeze()
    #     face_label = vertex_label[faces]
    #     face_mask = np.all(np.isin(face_label, selected_v_label), axis=1) ### slected
    
    faces_subset = faces[face_mask]
    
    edges_by_faces = faces_subset[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    edges = torch.sort(edges_by_faces, axis=1)[0].long()
    
    unique_rows, inverse_indices, counts = torch.unique(
        edges,
        dim=0,
        return_inverse=True,
        return_counts=True
    )
         
    mid_test = vertices[unique_rows].mean(axis=1)
    mid_idx_test = inverse_indices.reshape((-1, 3)) + len(vertices)
     
    f_test = torch.column_stack(
        [         
            faces_subset[:, 0],         
            mid_idx_test[:, 0],         
            mid_idx_test[:, 2],        
            mid_idx_test[:, 0],         
            faces_subset[:, 1],         
            mid_idx_test[:, 1],         
            mid_idx_test[:, 2],         
            mid_idx_test[:, 1],         
            faces_subset[:, 2],         
            mid_idx_test[:, 0],         
            mid_idx_test[:, 1],         
            mid_idx_test[:, 2]]     
    ).reshape((-1, 3))   
    
    
    new_faces = torch.vstack((faces[~face_mask], f_test))
    
    new_vertices = torch.vstack((vertices, mid_test))
        
    if vertex_attributes is not None:       
        new_attributes = {}       
        for key, values in vertex_attributes.items():
            
            if key == 'vertex_id':             
                attr_mid = values[unique_rows[:, 0]] # init (0 ~ 6889)
            elif key == 'vertex_label':
                attr_mid = values[unique_rows[:, 0]] # init (0 ~ 14)

            else:             
                attr_mid = values[unique_rows].mean(axis=1)

            new_attributes[key] = torch.concat((values, attr_mid))
    
    return new_vertices, new_faces, new_attributes


def collapse_edges(verts, verts_attr, selected_edges, faces, collapse_rate=0.5, smooth=False):
                
    collapse_map = torch.arange(len(verts), device=verts.device)
    vert_del = torch.zeros(len(verts), dtype=bool, device=verts.device) # False

    num_vert_include = torch.unique(selected_edges).shape[0]
    num_collapse = int(num_vert_include * collapse_rate)
    for i in range(num_collapse):
        if selected_edges.shape[0] == 0:
            print('No edge could be collapsed.')
            break
        
        # compute edge length and choose an edge to collapse
        edge_lengths = torch.linalg.norm(verts[selected_edges[:, 0]] - verts[selected_edges[:, 1]], axis=1)
        # min_edge_idx = torch.argmin(edge_lengths)
        max_edge_idx = torch.argmax(edge_lengths)
        
        v1, v2 = selected_edges[max_edge_idx]
        if vert_del[v1]:
            v1, v2 = v2, v1
            
        if vert_del[v1] and vert_del[v2]:
            print('Both end of an edge were deleted')

        # record replaced vertex
        collapse_map[collapse_map == v2] = v1 # keep v1

        # print(f'remove  {v2}:{~vert_del[v2]} <= {v1}: :{~vert_del[v2]}')
        # print('collapse_map', collapse_map)

        if smooth: ## TODO FIXME smooth verts
            # neighbors = edges[np.any((edges == 0) | (edges == 12), axis=-1)] # 注意这里edges并没有进行更新
            # neighbors = np.unique(collapse_map[neighbors])
            # print('neighbors', neighbors)
            # new_vert = np.mean(verts[neighbors], axis=0) ## neighbors
            # new_vert_1 = (verts[v1] + verts[v2]) / 2
            # print('diff', new_vert - new_vert_1)
            pass
        # elif midpoint:
        #     new_vert = (verts[v1] + verts[v2]) / 2
        #     new_vert_attr = (verts_attr[v1] + verts_attr[v2]) / 2
        else:
            new_vert = verts[v1]
            new_vert_attr = verts_attr[v1]
        
        # update vertices values
        verts[v1] = new_vert
        verts[v2] = new_vert

        vert_del[v2] = True
        
        verts_attr[v1] = new_vert_attr
        verts_attr[v2] = new_vert_attr
        
        # update selected edges
        selected_edges = selected_edges.clone() # RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location.
        selected_edges[selected_edges == v2] = v1
        selected_edges = selected_edges[(selected_edges[:, 0] != selected_edges[:, 1])]
        # selected_edges = selected_edges[np.any(~vert_del[selected_edges], axis=-1)]
        selected_edges = torch.unique(torch.sort(selected_edges, dim=1)[0], dim=0) # remove the redundant
        
    
    unique_vert, inverse_indices = torch.unique(collapse_map, return_inverse=True)
    # new_faces = inverse_indices[faces]
    new_faces = collapse_map[faces].clone()
    new_faces = torch.vstack([f for f in new_faces if len(torch.unique(f)) == 3]) # remove degenerate


    from trimesh.grouping import unique_rows
    sorted_faces = torch.sort(new_faces, dim=1)[0]
    unique, inverse = unique_rows(sorted_faces.cpu().numpy())
    unique_faces = new_faces[unique]

    # update vert_del
    selected_indices = torch.unique(unique_faces)
    selected_mask = torch.zeros(len(verts), dtype=bool)
    selected_mask[selected_indices] = True
    
    inverse_map = torch.full((verts.size(0),), -1, dtype=torch.long, device=verts.device)
    inverse_map[selected_indices] = torch.arange(selected_indices.size(0), device=verts.device)
    
    # print('inverse_map', inverse_map)

    new_faces = inverse_map[unique_faces]

    new_verts = verts[selected_mask]
    new_verts_attr = verts_attr[selected_mask]
    
    return new_verts, new_faces, new_verts_attr, ~selected_mask
