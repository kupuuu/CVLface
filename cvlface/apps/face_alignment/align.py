# RAFDB folder: /home/lh/new_disk/luhao/data/RAF/basic/Image/aligned

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
import cv2

sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

import argparse
from general_utils.huggingface_model_utils import load_model_by_repo_id
from general_utils.img_utils import visualize
from general_utils.img_utils import draw_ldmk
from general_utils.img_utils import prepare_text_img
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import torch
from general_utils.os_utils import get_all_files


def pil_to_input(pil_image, device='cuda'):
    # input is a rgb image normalized.
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(pil_image).unsqueeze(0).to(device)  # torch.randn(1, 3, 112, 112)
    return input


def deblack_border(pil_image, black_threshold=3):
    """Fill black edge bands by replicating nearest valid border pixels."""
    img = np.array(pil_image).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        return pil_image

    mask = np.all(img <= black_threshold, axis=2)
    if not mask.any():
        return pil_image

    h, w = mask.shape

    # Horizontal pass: fill left/right black runs in each row.
    for y in range(h):
        non_black = np.flatnonzero(~mask[y])
        if non_black.size == 0:
            continue
        left, right = int(non_black[0]), int(non_black[-1])
        if left > 0:
            img[y, :left] = img[y, left]
        if right < w - 1:
            img[y, right + 1:] = img[y, right]

    mask = np.all(img <= black_threshold, axis=2)

    # Vertical pass: fill top/bottom black runs in each column.
    for x in range(w):
        non_black = np.flatnonzero(~mask[:, x])
        if non_black.size == 0:
            continue
        top, bottom = int(non_black[0]), int(non_black[-1])
        if top > 0:
            img[:top, x] = img[top, x]
        if bottom < h - 1:
            img[bottom + 1:, x] = img[bottom, x]

    return Image.fromarray(img.astype(np.uint8))


def draw_aligned_ldmks_on_pil(pil_image, aligned_ldmks):
    """Draw normalized 5-point landmarks on one aligned PIL image."""
    if aligned_ldmks is None:
        return pil_image
    arr = np.array(pil_image).copy()
    if isinstance(aligned_ldmks, torch.Tensor):
        ldmk = aligned_ldmks[0].detach().cpu().numpy().reshape(-1)
    else:
        ldmk = np.asarray(aligned_ldmks[0]).reshape(-1)
    arr = draw_ldmk(arr, ldmk)
    return Image.fromarray(arr.astype(np.uint8))


def center_nose_on_image(pil_image, aligned_ldmks):
    """Shift aligned image so nose landmark (index 2) is at image center."""
    if aligned_ldmks is None:
        return pil_image, aligned_ldmks

    img = np.array(pil_image)
    h, w = img.shape[:2]

    if isinstance(aligned_ldmks, torch.Tensor):
        ldmks = aligned_ldmks.detach().cpu().clone()
        nose = ldmks[0, 2]
    else:
        ldmks = np.asarray(aligned_ldmks).copy()
        nose = ldmks[0, 2]

    nose_x = float(nose[0] * w)
    nose_y = float(nose[1] * h)
    target_x = w * 0.5
    target_y = h * 0.5

    dx = int(round(target_x - nose_x))
    dy = int(round(target_y - nose_y))

    if dx == 0 and dy == 0:
        return pil_image, aligned_ldmks

    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if isinstance(ldmks, torch.Tensor):
        ldmks[..., 0] = torch.clamp(ldmks[..., 0] + (dx / float(w)), 0.0, 1.0)
        ldmks[..., 1] = torch.clamp(ldmks[..., 1] + (dy / float(h)), 0.0, 1.0)
    else:
        ldmks[..., 0] = np.clip(ldmks[..., 0] + (dx / float(w)), 0.0, 1.0)
        ldmks[..., 1] = np.clip(ldmks[..., 1] + (dy / float(h)), 0.0, 1.0)

    return Image.fromarray(shifted.astype(np.uint8)), ldmks


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--aligner_id', type=str, default='minchul/cvlface_DFA_mobilenet')
    parser.add_argument('--data_root', type=str, default='/home/lh/new_disk/luhao/data/RAF/basic/Image/aligned/')
    parser.add_argument('--save_root', type=str, default='./example/aligned_images')
    parser.add_argument('--max_images', type=int, default=10,
                        help='Maximum number of images to process. Use -1 to process all images.')
    parser.add_argument('--deblack_method', type=str, default='none', choices=['none', 'border'],
                        help='Method to remove black edges after alignment.')
    parser.add_argument('--black_threshold', type=int, default=3,
                        help='Pixel threshold for black edge detection (0-255).')
    parser.add_argument('--nose_position_mode', type=str, default='center', choices=['template', 'center'],
                        help='template: keep original 5-point template position, center: move nose to image center.')
    args = parser.parse_args()

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf_token = os.getenv('HF_TOKEN', None)
    aligner = load_model_by_repo_id(repo_id=args.aligner_id,
                                    save_path=os.path.expanduser(f'~/.cvlface_cache/{args.aligner_id}'),
                                    HF_TOKEN=hf_token, ).to(device)

    data_root = os.path.abspath(args.data_root)
    save_root = os.path.abspath(args.save_root)

    all_image_paths = sorted(get_all_files(data_root, extension_list=['.jpg', '.png']))
    if args.max_images > 0:
        all_image_paths = all_image_paths[:args.max_images]

    for i, path in enumerate(all_image_paths):

        img1 = Image.open(path).convert('RGB')
        input1 = pil_to_input(img1, device)

        # align
        aligned_x1, orig_pred_ldmks1, aligned_ldmks1, score1, thetas1, normalized_bbox1 = aligner(input1)

        # save aligned images
        vis1 = visualize(aligned_x1.cpu().clone())

        draw_ldmks = aligned_ldmks1
        if args.nose_position_mode == 'center':
            vis1, draw_ldmks = center_nose_on_image(vis1, aligned_ldmks1)

        if args.deblack_method == 'border':
            vis1 = deblack_border(vis1, black_threshold=args.black_threshold)

        # Draw landmarks after any post-process to keep point locations visually consistent.
        vis2 = draw_aligned_ldmks_on_pil(vis1, draw_ldmks)

        rel_path = os.path.relpath(path, data_root)
        save_path = os.path.join(save_root, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis1.save(save_path)
        vis2.save(os.path.splitext(save_path)[0] + '_ldmks.png')
