from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from addict import Dict
import numpy as np
import math
from libs.autoencoder import encoder_res50
import torch
from libs.utils import (
    to_tensor,
    NLM_woSoft,
    transform_topk,
    draw_labelmap_np,
    color_normalize,
    sorted_alphanumeric,
)
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import shutil
from torch.optim import Optimizer, Adam, lr_scheduler
import sys

torch.autograd.set_detect_anomaly(True)
COLORS = {
    0: [0, 0, 1.0],
    1: [0, 1.0, 0],
    2: [1.0, 0, 0],
    3: [1.0, 1.0, 0],
    4: [1.0, 1.0, 1.0],
}


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--crop_size", default=480, type=int, help="Size to crop images to (square)",
    )
    parser.add_argument(
        "--sigma", default=0.5, type=float, help="std for gaussian at coordinates",
    )
    parser.add_argument("--erase", action="store_true", help="erase output directory")

    return parser.parse_args()


def get_bodypart_channels(levels):
    """Gets bodyparts and associates them with a channel

    Args:
        levels ([Pandas df.index.levels]): [The levels of a dataframe's index]
    """

    channels_dict = {}

    for i, bodypart in enumerate(levels[0]):
        channels_dict[bodypart] = i

    return Dict(channels_dict)


def get_bodypart_coords(row_data, levels):
    """Gets bodyparts and associates them with a channel

    Args:
        levels ([Pandas df.index.levels]): [The levels of a dataframe's index]
    """

    coord_dict = {}

    for i, bodypart in enumerate(levels[0]):
        x, y = row_data["mac"][bodypart].array
        coord = [x, y]
        coord_dict[bodypart] = coord

    return Dict(coord_dict)


def draw_point(image, x, y, r=10):
    draw = ImageDraw.Draw(image)

    if isinstance(x, list) and isinstance(y, list):
        for i in range(len(x)):
            draw.ellipse((x[i] - r, y[i] - r, x[i] + r, y[i] + r), fill=(255, 0, 0, 0))
    elif isinstance(x, float) and isinstance(y, float):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, 0))
    else:
        raise ValueError(f"x is of type {type(x)} and y is {type(y)}")

    image.show()


def collect_data(data):
    channel_dict = None
    coord_dict = None
    data_dict = {}

    for i in tqdm(data.iterrows()):
        # Get name and data
        filename, row_data = i
        filename = Path(filename).name

        img_path = data_dir / filename

        levels = row_data["mac"].index.levels
        if channel_dict is None:
            channels_dict = get_bodypart_channels(levels)

        coord_dict = get_bodypart_coords(row_data, levels)
        single_data = Dict({"path": img_path, "data": coord_dict})
        img_num = int(str(single_data.path.stem).replace("img", ""))
        # data_list.append(single_data)
        data_dict[img_num] = single_data

    return channels_dict, data_dict


def make_lbls(coords, lbl_size, sigma):
    lbls = np.zeros((lbl_size, lbl_size, coords.shape[1]))
    # For each coordinate
    for i in range(coords.shape[1]):
        if math.isnan(coords[0, i]) or math.isnan(coords[1, i]):
            continue
        if sigma > 0:
            [x, y] = coords[:, i]
            draw_labelmap_np(lbls[:, :, i], [x, y], sigma)
        else:
            lbls[int(coords[0, i]), int(coords[1, i]), i] = 1.0

    return lbls


def save_seg_img(image, lbl, name="./frame.jpg"):
    """Overlay label onto image and save

    Args:
        image ([numpy array]): [image to be overlayed]
        lbl ([numpy array]): [label to overlay, shape is (N,N,C)]
        name (str, optional): [filename to save]. Defaults to "frame.png".
    """

    image = image.copy()
    # If we have a non-binary label
    lbl_image = np.zeros((lbl.shape[0], lbl.shape[1], 3))
    if len(np.unique(lbl)) > 2:
        # This will be a heatmap
        for i in range(lbl.shape[-1]):
            # print("i: ", i)
            lbl_channel = np.expand_dims(lbl[:, :, i], axis=-1)

            lbl_channel = np.tile(lbl_channel, (1, 1, 3))

            # lbl_channel[lbl_channel < 0.005] = 0
            lbl_channel = lbl_channel * COLORS[i]

            lbl_image += lbl_channel
        lbl_image = cv2.normalize(
            lbl_image,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        image = np.expand_dims(image, axis=-1)
        image = np.tile(image, (1, 1, 3))

        image = cv2.normalize(
            image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        image = cv2.addWeighted(lbl_image, 0.8, image, 0.2, 0, dtype=cv2.CV_32F)
        plt.imsave(name, image)


def get_coords(channel_dict, seg_coords_dict):
    coords = np.zeros(shape=(len(channel_dict.keys()), 2))
    for bodypart in seg_coords_dict.keys():
        [x, y] = seg_coords_dict[bodypart]
        coords[channel_dict[bodypart]] = [x, y]

    coords = np.transpose(coords)
    return coords


if __name__ == "__main__":

    # data_path = Path(
    #    "/Users/Sunsmeister/Desktop/Research/Brain/DGP/dgp_propagate/data/reachingvideo1/CollectedData_Mackenzie.h5"
    # )
    data_path = Path(
        "./data/reach/raw_data/" + "labeled-data/reachingvideo1/CollectedData_mac.h5"
    )
    frame_data_path = Path("./video_frames")

    args = parsed_args()

    crop_size = args.crop_size
    sigma = args.sigma

    write_dir = Path("./prop_outputs")
    if args.erase:
        shutil.rmtree(write_dir, ignore_errors=True)
    write_dir.mkdir(exist_ok=True)

    to_tensor = to_tensor()

    # -----------------------
    # -----  Load data  -----
    # -----------------------

    data_dir = data_path.parent
    data = pd.read_hdf(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_frames = os.listdir(frame_data_path)
    video_frames = sorted_alphanumeric(video_frames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Get channel idx and keypoint data
    channel_dict, data_dict = collect_data(data)
    data_list = list(data_dict.values())
    data_list.sort(key=lambda x: x.path.name)

    # Read reference frame
    ref_frame_data = data_list[0]
    # Read in grayscale mode
    ref_image = cv2.imread(str(ref_frame_data.path), 0)
    og_h, og_w = ref_image.shape

    # Read in coordinates:
    ref_coords = get_coords(channel_dict, ref_frame_data.data)
    # Scale coords to feature size
    ref_coords_featsize = np.zeros(ref_coords.shape)
    ref_coords_featsize[0, :] = ref_coords[0, :] * float(crop_size) / float(og_w) / 8.0
    ref_coords_featsize[1, :] = ref_coords[1, :] * float(crop_size) / float(og_h) / 8.0

    # ---------------------------------
    # -----  Create segmentation  -----
    # ---------------------------------
    # Make label (seg map) --> (H,W,C)
    lbl_size = int(crop_size / 8.0)
    # NOTE Use featsize coords --> This is the size of the encoded feats
    ref_lbls_featsize = make_lbls(ref_coords_featsize, lbl_size, sigma)
    # transform label into torch
    ref_lbls_featsize_tensor = to_tensor(ref_lbls_featsize).unsqueeze(0).double()

    # Plot lbls
    # for i in range(ref_lbls_featsize.shape[2]):
    #    plt.imshow(ref_lbls_featsize[:, :, i], cmap="gray")
    #    plt.show()

    # --------------------------------------
    # -----  Transform reference frame -----
    # --------------------------------------

    # Resize image
    ref_image_cropsize = cv2.resize(ref_image, dsize=(crop_size, crop_size))

    # Make lbls at crop_size for visualization
    ref_coords_cropsize = np.zeros(ref_coords.shape)
    ref_coords_cropsize[0, :] = ref_coords[0, :] * float(crop_size) / float(og_w)
    ref_coords_cropsize[1, :] = ref_coords[1, :] * float(crop_size) / float(og_h)
    ref_lbls_cropsize = make_lbls(ref_coords_cropsize, crop_size, sigma)

    # Save image + labels at crop size
    save_seg_img(
        ref_image_cropsize,
        ref_lbls_cropsize,
        write_dir / Path("./reference_label_cropsize.jpg"),
    )

    # -----------------------
    ref_image_cropsize_tensor = to_tensor(ref_image_cropsize).unsqueeze(0)
    # Shape is now B, C, H, W
    if ref_image_cropsize_tensor.shape[1] == 1:
        ref_image_cropsize_tensor = ref_image_cropsize_tensor.repeat(1, 3, 1, 1)

    ref_image_cropsize_tensor = color_normalize(ref_image_cropsize_tensor, mean, std)

    # -----  Get model  -----
    # -----------------------

    # Get resnet model
    model = encoder_res50().to(device)

    # ----------------------------
    # -----  Setup affinity  -----
    # ----------------------------

    # Get affinity computation function
    nlm = NLM_woSoft()
    softmax = nn.Softmax(dim=1)
    end_softmax = nn.Softmax(dim=-1)
    temp = 1
    k = 5

    # -----------------------------
    # -----  Setup optimizer  -----
    # -----------------------------

    opt = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    l1_loss = torch.nn.L1Loss()
    # ------------------------------
    # -----  Propagate labels  -----
    # ------------------------------
    global_count = 0
    save_iter = 5
    while True:
        # Propagate reference frame to every other frame
        for count, target_frame in enumerate(tqdm(video_frames)):
            img_num = int(target_frame.split(".")[0])

            ref_feat = model(ref_image_cropsize_tensor)
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            # ----------------------------------
            # -----  Read image and label  -----
            # ----------------------------------
            target_image = cv2.imread(str(frame_data_path / target_frame), 0)
            og_h, og_w = target_image.shape
            target_image_cropsize = cv2.resize(
                target_image, dsize=(crop_size, crop_size)
            )
            target_image_cropsize_tensor = to_tensor(target_image_cropsize).unsqueeze(0)
            # Shape is now B, C, H, W
            if target_image_cropsize_tensor.shape[1] == 1:
                target_image_cropsize_tensor = target_image_cropsize_tensor.repeat(
                    1, 3, 1, 1
                )
            target_image_cropsize_tensor = color_normalize(
                target_image_cropsize_tensor, mean, std
            )

            # --------------------------
            # -----  Get features  -----
            # --------------------------
            #  Get feature
            target_feat = model(target_image_cropsize_tensor)
            target_feat = F.normalize(target_feat, p=2, dim=1)

            # -----------------------------
            # -----  Propagate label  -----
            # -----------------------------
            # Compute affinity
            aff = nlm(ref_feat, target_feat)
            aff = softmax(aff * temp).double()

            pred_lbls_featsize = transform_topk(aff, ref_lbls_featsize_tensor, k=k)

            # --------------------------------
            # -----  Normalize channels  -----
            # --------------------------------

            """
            # For each channel
            for t in range(len(channel_dict.keys())):
                # If there is no label in this channel
                if pred_lbls_featsize[:, t, :, :].sum() == 0:
                    continue

                # Normalize channel
                pred_lbls_featsize[:, t, :, :] = (
                    pred_lbls_featsize[:, t, :, :] - pred_lbls_featsize[:, t, :, :].min()
                )
                pred_lbls_featsize[:, t, :, :] = (
                    pred_lbls_featsize[:, t, :, :] / pred_lbls_featsize[:, t, :, :].max()
                )
            """

            # ---------------------------------
            # -----  Upsample pred label  -----
            # ---------------------------------
            pred_lbls_cropsize_tensor = (
                torch.nn.functional.interpolate(
                    pred_lbls_featsize, scale_factor=(8, 8), mode="bilinear"
                )
                .squeeze()
                .permute(1, 2, 0)
            )
            # pred_lbls_cropsize_tensor = end_softmax(pred_lbls_cropsize_tensor)
            # --------------------------------
            # -----  pred label to numpy  -----
            # --------------------------------
            pred_lbls_cropsize_numpy = pred_lbls_cropsize_tensor.detach().numpy()

            if global_count % save_iter == 0:
                save_seg_img(
                    target_image_cropsize,
                    pred_lbls_cropsize_numpy,
                    write_dir / Path(f"./predicted_label_cropsize_{count}.jpg"),
                )

            # --------------If target label---------------------------

            if img_num in data_dict:

                target_frame = data_dict[img_num]
                target_coords = get_coords(channel_dict, target_frame.data)

                # Initialize target coords
                target_coords_cropsize = np.zeros(target_coords.shape)
                target_coords_featsize = np.zeros(target_coords.shape)

                # Scale coords
                target_coords_cropsize[0, :] = (
                    target_coords[0, :] * float(crop_size) / float(og_w)
                )
                target_coords_cropsize[1, :] = (
                    target_coords[1, :] * float(crop_size) / float(og_h)
                )

                target_coords_featsize[0, :] = (
                    target_coords[0, :] * float(crop_size) / float(og_w) / 8.0
                )
                target_coords_featsize[1, :] = (
                    target_coords[1, :] * float(crop_size) / float(og_h) / 8.0
                )

                # Make labels

                target_lbls_cropsize = make_lbls(
                    target_coords_cropsize, crop_size, sigma
                )
                target_lbls_featsize = make_lbls(
                    target_coords_featsize, lbl_size, sigma
                )

                target_lbls_cropsize_tensor = to_tensor(target_lbls_cropsize).permute(
                    1, 2, 0
                )

                target_lbls_featsize_tensor = to_tensor(target_lbls_featsize).unsqueeze(
                    0
                )

                """
                save_seg_img(
                    target_image_cropsize,
                    target_lbls_cropsize,
                    write_dir / Path(f"./target_label_cropsize_{count}.jpg"),
                )
                loss = l1_loss(pred_lbls_cropsize_tensor, target_lbls_cropsize_tensor)
                loss.backward(retain_graph=True)
                print("loss: ", loss)
                opt.step()
                opt.zero_grad()
                """

                ref_lbls_featsize_tensor = target_lbls_featsize_tensor
                ref_image_cropsize_tensor = target_image_cropsize_tensor

            # else:
            #    ref_lbls_featsize_tensor = pred_lbls_featsize
            #    ref_image_cropsize_tensor = target_image_cropsize_tensor

        global_count += 1
