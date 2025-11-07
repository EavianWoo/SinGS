import os
import json
import glob
import tqdm
import argparse

import cv2
import numpy as np
# from segment_anything_2.sam2.build_sam
from sam2.build_sam import build_sam2_video_predictor 

# sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml" # ./sam2_configs/
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


# import matplotlib.pyplot as plt
# def show_mask(mask, ax, obj_id=None, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         cmap = plt.get_cmap("tab10")
#         cmap_idx = 0 if obj_id is None else obj_id
#         color = np.array([*cmap(cmap_idx)[:3], 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=200):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def resize_bbox(box, img_width=1080, img_height=1080, scale_factor=1.2):
    x1, y1, x2, y2 = box
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    scaled_width = bbox_width * scale_factor
    scaled_height = bbox_height * scale_factor
    
    center_x = x1 + bbox_width / 2
    center_y = y1 + bbox_height / 2
    
    scaled_x1 = center_x - scaled_width / 2
    scaled_y1 = center_y - scaled_height / 2
    scaled_x2 = center_x + scaled_width / 2
    scaled_y2 = center_y + scaled_height / 2
    
    scaled_x1 = int(max(0, scaled_x1))
    scaled_y1 = int(max(0, scaled_y1))
    scaled_x2 = int(min(img_width, scaled_x2))
    scaled_y2 = int(min(img_height, scaled_y2))
    
    scaled_box = np.array((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
    
    return scaled_box


def check_frames(video_dir):
    '''Read frames from video directory and return frame names, image height and width.'''
    
    # `video_dir` for SAM2 should be a directory of JPEG frames with filenames like `<frame_index>.jpg`
    files = os.listdir(video_dir)
    jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
    png_files = [f for f in files if f.lower().endswith('.png')]

    if len(jpg_files) == 0:
        assert len(png_files) > 0, f'No jpg or png file found in video directory.'
        
        # if there only exists png file then create jpg copies.
        # as SAM2 only check jpg file.
        print('Copy temp jpg image...')
        for png_file in png_files:
            img = cv2.imread(os.path.join(video_dir, png_file))
            copy_path = os.path.join(video_dir, png_file.replace('.png', '.jpg'))
            cv2.imwrite(copy_path, img)
        
        frame_names = sorted([f.replace('.png', '.jpg') for f in png_files])

        # or just alter sam2/utils/misc.py to accept png
        
    else:
        frame_names = sorted(jpg_files)

    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    IMG_H = first_frame.shape[0]
    IMG_W = first_frame.shape[1]
    
    return frame_names, IMG_H, IMG_W
    

def check_poses(keypoints_path):
    '''Build points and box prompt from AlphaPose keypoint json file.'''
    with open(keypoints_path, 'r') as f:
        poses = json.load(f)

    # Extract keypoints and box, number the people
    # use halp26 
    ### NOTE # Alphapose ###
    # Ensure set --pose_track if more than one person in the video,
    # or the idx in the keypoint json file is meaningless (default 0 for everyone).
    ### NOTE # Alphapose ###
    # Never use --pose_track when there is only one person in the video,
    # or the idx in the keypoint json file is also meaningless (maybe more than one unique number).
    pose_prompt = {}
    for i, pose in enumerate(poses):
        person_id = pose['idx'] if isinstance(pose['idx'], int) else pose['idx'][0]
        # Alphapose BUG
        while isinstance(person_id, list):
            person_id = person_id[0]
        if person_id not in pose_prompt:
            pose_prompt[person_id] = []
        
        pose_prompt[person_id].append(
            {
                'image_id': pose['image_id'],
                'keypoints': pose['keypoints'],
                'box': pose['box'],
            }
        )
        
    return pose_prompt


def main(args):
    video_dir = args.video_dir
    keypoints_path = args.keypoints_path
    only_first_frame = args.only_first_frame
    use_box = args.use_box
    resize_box = args.resize_box
    eorde_size = args.eorde_size

    frame_names, IMG_H, IMG_W = check_frames(video_dir)
    
    pose_prompt = check_poses(keypoints_path)

    num_person = len(pose_prompt.keys())
    assert num_person == 1, f'Only supprt single person for now, find more than one person in keypoint json file'

    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')# device=device)
    
    # Initailize the inference state
    inference_state = predictor.init_state(video_path=video_dir)
    # predictor.reset_state(inference_state)
    
    # add keypoints guidance on assigned (the 1st frame)/ each frame 
    ann_obj_id = 1  #  the number we interact with (it can be any person_id)
    # ann_obj_id = person_id
    for ann_obj_id in pose_prompt.keys():
        for ann_frame_idx, pose in enumerate(pose_prompt[ann_obj_id]):

            points = np.array(pose['keypoints']).reshape(-1, 3)[:, :2]
            
            # ignore face and feet
            selected_points = points[5:20] # points #
            
            # # remove low confidence
            # selected_idx = selected_points[:, -1] > 0.5
            # selected_points = selected_points[selected_idx]
            
            # face_slice = slice(0, 5)
            # feet_slice = slice(20, 24)
            # heel_slice = slice(25, 27)
            # selected_idx = list(range(26))
            
            refer_to_heel = True
            if refer_to_heel:
                points_heel = points[-2:]
                selected_points = np.concatenate([selected_points, points_heel], axis=0)
                
            # # add some points
            enable_additional_points = True
            additional_points = []
            if enable_additional_points:
                # center
                additional_points.append((points[17] + points[18]) / 2.)
                # 
                additional_points.append((points[4] + points[17]) / 2.)
                additional_points.append((points[5] + points[17]) / 2.)
                additional_points = np.stack(additional_points, axis=0)

                selected_points = np.concatenate([selected_points, additional_points], axis=0)
            
            # disable box
            if use_box:
                box = np.array(pose['box'])
                if resize_box:
                    box = resize_bbox(box, img_height=IMG_H, img_width=IMG_W)

            labels = np.ones(len(selected_points))
            # Step 2 add positive points and labels
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx, # the frame index we attach prompts to
                obj_id=ann_obj_id, # a unique id to object, use persion id
                points=selected_points,
                labels=labels, # a list of which length = point number
                # box=box,
            )

            # Experiments demonstrate that just enabling the points and bbox prompt from the 1st frame gets better segmentation.
            # More information can distract the propogation.
            if only_first_frame and ann_frame_idx > 0:
                break

    # Step 3 propogate the prompts to get masklet across the video
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    # Step 4 Read and save masks 
    root_dir = os.path.dirname(video_dir)
    mask_dir = os.path.join(root_dir, args.out_dir_name)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
    print('Save masks in ', mask_dir)

    for out_frame_idx in tqdm.tqdm(range(len(frame_names))): 
        for out_obj_ids, out_mask in video_segments[out_frame_idx].items():
            # assert out_obj_ids == 0, f'not support multi-person for temporary!'
            # mask_name = pose_prompt[out_obj_ids][out_frame_idx]['image_id']
            mask_name = f'{out_frame_idx:06d}.png'
            mask_path = os.path.join(mask_dir, mask_name)

            out_mask = (out_mask * 255).transpose(1, 2, 0)
            if eorde_size > 0:
                kernel_erode = np.ones((eorde_size, eorde_size), np.uint8)
                out_mask = cv2.erode(out_mask, kernel_erode, iterations=1)
            cv2.imwrite(mask_path, out_mask)

    
    # Last Step: remove temp jpg
    print('Remove temp jpg image...')
    for jpg_file in glob.glob(os.path.join(video_dir, '*.jpg')):
        os.remove(jpg_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dir', type=str, help='The directory to store all the frames.',
                        default="/media/cxh/cxh/513/AlphaPose/examples/male-3-casual/images")
    parser.add_argument('-k', '--keypoints_path', type=str, help='The path for AlphaPose results.',
                        default='/media/cxh/cxh/513/AlphaPose/examples/res/alphapose-results.json')
    parser.add_argument('-o', '--out_dir_name', type=str, default='masks', help='The name of the directory to store masks.')
    parser.add_argument('--only_first_frame', type=bool, default=True, help='Only add keypoints and bbox prompt from the 1st frame. -> Better results')
    parser.add_argument('--use_box', action="store_true", help='Whether to use the box from AlphaPose.')

    parser.add_argument('--resize_box', type=bool, default=True, help='Whether to resize the box from AlphaPose to provide better prompt for segmentation. -> Better results')
    parser.add_argument('--eorde_size', type=int, default=1, help='The kernel size of erosion performed on the mask')
    parser.add_argument('--save_masked', type=str, default=True, help='Whether to save masked image for quality validation.')
    parser.add_argument('--save_posed_video') # # TODO save video
    parser.add_argument('--save_masked_video') # TODO save video
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
