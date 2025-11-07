# Adapted from https://github.com/apple/ml-hugs/blob/main/hugs/renderer/gs_renderer.py

import math
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)


def get_render_pkg(
    data, 
    human_gs_out,
    bg_color, 
    scaling_modifier=1.0, 
):

    feats = None
    feats = human_gs_out['shs']
    means3D = human_gs_out['xyz']
    opacity = human_gs_out['opacity']
    scales = human_gs_out['scales']
    rotations = human_gs_out['rotq']
    active_sh_degree = human_gs_out['active_sh_degree']
    
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )
    
    render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
    render_pkg['human_radii'] = render_pkg['radii']
        
    return render_pkg
    
    
def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    
    raster_settings = GaussianRasterizationSettings(
        image_height= int(data['image_height']),
        image_width= int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    visibility_filter = radii > 0
    
    render_res = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : visibility_filter,
        "radii": radii,
    }
    
    ### for parsing mask
    # Use visibility filter to find visible Gaussians
    if 'parse' not in data.keys():
        return render_res
    
    if False:
        # selected optimiztion ### TODO
        selected_mask = data['parse']
        
        # Projected coordinates of visible Gaussians
        means3D_h = torch.cat([means3D, torch.ones(means3D.shape[0], 1, device=means3D.device)], dim=-1)  # (N, 4)
        means2D_h = torch.matmul(means3D_h, data['full_proj_transform']) # (N, 4)
        projected_points = means2D_h[:, :2] / means2D_h[:, 3:4]
        visible_gaussians_2D = projected_points[visibility_filter]
        
        # Filter based on hand mask (convert 2D coordinates to image space)
        image_height, image_width = data['image_height'], data['image_width']
        visible_gaussians_2D[:, 0] = (visible_gaussians_2D[:, 0] + 1) * (image_width / 2)
        visible_gaussians_2D[:, 1] = (visible_gaussians_2D[:, 1] + 1) * (image_height / 2)
        
        # Find Gaussians that are inside the hand mask
        selected_gaussians_filter = selected_mask[visible_gaussians_2D[:, 1].long(), visible_gaussians_2D[:, 0].long()].long()
        selected_gaussians_indices = visibility_filter.nonzero(as_tuple=False).squeeze()[selected_gaussians_filter == 1]

        
    if False:
        selected_means3D = means3D[selected_gaussians_indices]
        selected_feats = feats[selected_gaussians_indices]
        selected_opacity = opacity[selected_gaussians_indices]
        selected_scales = scales[selected_gaussians_indices]
        selected_rotations = rotations[selected_gaussians_indices]
    
        # Rasterize the selected Gaussians (hand) to generate a new image
        selected_render_image, _ = rasterizer(
            means3D=selected_means3D,
            means2D=means2D[selected_gaussians_indices],  # We reuse the previously computed 2D projection
            shs=shs if shs is not None else selected_feats,  # Use selected hand features
            opacities=selected_opacity,
            scales=selected_scales,
            rotations=selected_rotations,
            colors_precomp=selected_feats if shs is None else None
        )

        selected_rendered_image = torch.clamp(selected_render_image, 0.0, 1.0)
        
        render_res.update({
            "selected_gaussians": selected_gaussians_indices, ### TODO
            "selected_rendered_image": selected_rendered_image,
        })
    
    return render_res

