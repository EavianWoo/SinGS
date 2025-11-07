import numpy as np
from math import exp

import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable

from pytorch3d.ops import knn_points
from pytorch3d.ops import laplacian, cot_laplacian, norm_laplacian

from sings.rec.utils.body_model.smpl_parsing import parse_weights


class L2Norm(torch.nn.Module):
    def __init__(self, 
        lambda_xyz_offsets=0.005,
        lambda_scales_diff=0.005,
        lambda_max_scale=0.001,
        max_scale_threshold=0.008, 
        lambda_min_opacity=0.0001,
        min_opacity_threshold=0.2, 
    ):
        super().__init__()
        
        self._lambda_xyz_offset = lambda_xyz_offsets
        self._lambda_scales_diff = lambda_scales_diff
        self._lambda_max_scale = lambda_max_scale
        self._max_scale_threshold = max_scale_threshold
        self._lambda_min_opacity = lambda_min_opacity
        self._min_opacity_threshold = min_opacity_threshold

        
    def forward(self, human_gs_out):
        xyz_offsets = human_gs_out['xyz_offsets']
        scales = human_gs_out['scales'][:, 0]
        scales_diff = scales - scales.mean(dim=0)

        # roundness = scales[:, 0] - scales[:, 1]
        # thickness = scales[:, -1]
        
        # limit max
        scale_thresh_idxs = (scales > self._max_scale_threshold)

        loss = self._lambda_xyz_offset * (xyz_offsets).norm() + \
               self._lambda_scales_diff * (scales_diff).norm() + \
               self._lambda_max_scale * (scales[scale_thresh_idxs]).norm()
        
        if 'opacity' in human_gs_out:
            opacity = human_gs_out['opacity']
            opacity_thresh_idx = (opacity < self._min_opacity_threshold)
            loss += self._lambda_min_opacity * (0.5 - opacity[opacity_thresh_idx]).norm()
        
        return loss
    

class GaussiansEdgeLoss(torch.nn.Module):
    """Encourage gaussians to be more close together, and anchor surface more smooth"""
    """Now only works for isotropic gaussians"""
    def __init__(self, K=9, eps=1e-12):
        super().__init__()
        self._K = K
        self._eps = eps
        pass
    
    def forward(self, human_gs_out):
        verts =  human_gs_out['xyz_canon']
        scales = human_gs_out['scales'][:, 0] # isotropic [:, 0]
        # rot = human_gs_out['rotmat']
        # edges = build_edges(verts, self._K)
        
        
        dists_knn, idx_knn, _ = knn_points(verts.unsqueeze(0), verts.unsqueeze(0), K=self._K) # shape (1, N, K+1), (1, N, K+1) 其中idx[..., 0]是自身 邻接点idx[..., 1:]

        ### TODO consider gaussian size      
        edge_vectors = verts[idx_knn[0, :, 1:]] - verts.unsqueeze(1) # -> (N, K, 3)
        edge_lengths = torch.norm(edge_vectors, dim=-1).mean(dim=-1, keepdim=True).detach()  # (N, K)

        scale_proj_i = scales.unsqueeze(1) # (N, 1)
        scale_proj_j = scales[idx_knn[0, :, 1:]] # (N, K)
        
        # Scale-Edge Loss
        len_factor = 1.0
        loss = ((scale_proj_i - len_factor * edge_lengths) ** 2).mean()
    
        limit_length = False
        if limit_length:
            edge_loss = edge_lengths.mean() # ((edge_lengths - edge_lengths.mean()) ** 2).mean() # 
            loss += edge_loss
        return loss


class RegionLaplacianLoss_v2(torch.nn.Module):
    def __init__(self, verts, edges, vertex_labels, faces=None, region_weights=None, laplacian_type="standard"):  # 'standard', ' cotangent', 'norm' 
        """only for unique edges"""

        super().__init__()
        
        # if isinstance(vertex_labels, np.ndarray):
        #     vertex_labels = torch.from_numpy(vertex_labels).to('cuda')
        # self.vertex_labels = vertex_labels
        
        self.unique_labels = torch.unique(vertex_labels)
        # edges, _ = grouping.unique_rows(edges) # ensure unique_edges
        
        # self.edge_label = self.vertex_labels[edges]
        if laplacian_type == "standard":
            self.laplacian_fn = laplacian
        elif laplacian_type == "cotangent":
            self.laplacian_fn = cot_laplacian
        elif laplacian_type == "norm": ### TODO
            raise NotImplementedError
            self.laplacian_fn = norm_laplacian
        else:
            raise NotImplementedError
        
        
        self.reset_laplacians(verts, edges, vertex_labels, faces)

        self.weights = parse_weights(region_weights)

    
    def reset_laplacians(self, verts, edges, vertex_labels, faces=None): # faces used for cotangent laplacians

        # standard laplacian is stable
        # cot or norm version may be affected as the vertices updation
        if self.laplacian_fn == cot_laplacian:
            assert faces is not None, f'faces is supposed to be provided when using cotangent laplacian.'

        if isinstance(vertex_labels, np.ndarray):
            vertex_labels = torch.from_numpy(vertex_labels).to('cuda')
        self.vertex_labels = vertex_labels
        
        self.edge_label = self.vertex_labels[edges]
        
        self.laplacians = []
        self.vertex_partitions = [] # overlapped
        for label in self.unique_labels:
            if self.laplacian_fn == laplacian:
                verts_idx_included = (self.vertex_labels == label)
                selected_verts = verts[verts_idx_included]
                selected_edges = edges[torch.all(self.edge_label == label, dim=1)] # with the same label
                
                ### global index -> local index
                ### selected_edges = torch.sort(selected_edges, dim=-1) # unique guarantee has been moved outside
                ### ensure the edges are unique, or error occurs when reindexing
                unique_verts, inverse_indices = torch.unique(selected_edges, return_inverse=True)
                part_edge_local = inverse_indices.reshape(selected_edges.shape)

                with torch.no_grad():
                    L = self.laplacian_fn(selected_verts, part_edge_local)
                    ## cot_laplacian / norm_laplacian
            
            elif self.laplacian_fn == cot_laplacian:
                
                self.face_label = self.vertex_labels[faces]
                selected_faces = faces[torch.any(self.face_label == label, dim=1)]
                
                verts_idx_included = torch.unique(selected_faces)
                selected_verts = verts[verts_idx_included]
                
                unique_verts, inverse_indices = torch.unique(selected_faces, return_inverse=True)
                part_face_local = inverse_indices.reshape(selected_faces.shape)
                with torch.no_grad():
                    L, _ = self.laplacian_fn(selected_verts, part_face_local)
                    
                    
            self.laplacians.append(L)
            self.vertex_partitions.append(verts_idx_included)
    
    
    def forward_hands(self, x, hand_strength=1000):
        loss = 0.
        for i in [6, 7]:
            # x_part = x[self.vertex_labels == i]
            x_part = x[self.vertex_partitions[i]]
            x_part = torch.matmul(self.laplacians[i], x_part)
            loss += hand_strength * x_part.pow(2).mean()
            
        return loss  
    
     
    def forward(self, x):
        loss = 0.
        for i in self.unique_labels:
            
            # x_part = x[self.vertex_labels == i]
            x_part = x[self.vertex_partitions[i]]
            x_part = torch.matmul(self.laplacians[i], x_part)
            loss += self.weights[i] * x_part.pow(2).mean()
            
        return loss



### Laplacian smoothing based on KNN ###
def build_edges(verts, K=9):
    knn_out = knn_points(verts[None], verts[None], K=K+1) ###
    knn_idx = knn_out.idx.squeeze(0)[:, 1:]
    
    indices = torch.arange(verts.shape[0], device=verts.device).unsqueeze(1)
    edges = torch.cat([indices.repeat(1, K).reshape(-1, 1), knn_idx.reshape(-1, 1)], dim=1)
    
    return edges

    
def pcd_laplacian_smoothing(verts, edges, method: str = "uniform"):
    
    with torch.no_grad():
        L = laplacian(verts, edges)
        
    loss = L.mm(verts)
    loss = loss.norm(dim=1)

    return loss.mean()
    

class LaplacianSmoothing(torch.nn.Module):
    def __init__(self, K=9):
        super().__init__()
        self._K = K
    
    def forward(self, smooth_dict, edges=None): # human_gs_out):
        loss = 0.
        ## TODO knn tracking
        # TODO normal smoothing
        
        if edges is None:
            verts = smooth_dict['xyz_canon'] 
            edges = build_edges(verts, self._K)
        for _, verts in smooth_dict.items(): # xyz, xyz_canon, color, rotation, opacity
            loss += pcd_laplacian_smoothing(verts, edges)

        return loss
        

    