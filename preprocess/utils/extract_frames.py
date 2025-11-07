import os
import argparse
import tqdm

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, help='The path of video to be processed.')
    parser.add_argument('-o', '--out_dir', type=str, help='The folder to store the images.')
    parser.add_argument('--frame_interval', type=int, default=1, help='Control the interval of frames to be saved. \n Default is 1 (save every frame).')
    
    parser.add_argument('--img_center', default=None)
    parser.add_argument('--img_h', default=None)
    parser.add_argument('--img_w', default=None)
    args = parser.parse_args()

    return args


def main(args):
    
    img_dir = os.path.join(args.out_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)

    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total Frame:', frame_cnt)
    
    frame_interval = args.frame_interval
    print('Set frame interval', frame_interval)
    print('Start extracting...')

    # extract
    img_cnt = 0
    for i in tqdm.trange(frame_cnt):
        ret, frame = cap.read()
        if i % frame_interval:
            continue

        if not ret:
            break
        # frame = cv2.undistort(frame, K, dist_coeffs)


        if args.img_center: # if crop
            center_x = args.img_center[0]
            center_y = args.img_center[1]
            w = args.img_w
            h = args.img_h
            cropped_image = cv2.getRectSubPix(frame, (w, h), (center_x, center_y))
            resized_image = cv2.resize(cropped_image, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)
            frame = resized_image

        img_path = os.path.join(img_dir, f'{img_cnt:06d}.png')
        img_cnt += 1
        
        cv2.imwrite(img_path, frame)
    
    print(f'Saved frames in {args.out_dir}.')

    cap.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)