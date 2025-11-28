from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine, get_warp_matrix


def bbox_xyxy2cs_torch(
    bbox: torch.Tensor, padding: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (Tensor): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - torch.Tensor[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - torch.Tensor[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.dim()
    if dim == 1:
        bbox = bbox.unsqueeze(0)

    # get bbox center and scale
    x1, y1, x2, y2 = torch.chunk(bbox, 4, dim=1)
    center = torch.cat([x1 + x2, y1 + y2], dim=1) * 0.5
    scale = torch.cat([x2 - x1, y2 - y1], dim=1) * padding

    if dim == 1:
        center = center.squeeze(0)
        scale = scale.squeeze(0)

    return center, scale

def top_down_affine_tensor_torch(input_size: dict, 
                          bbox_scale: torch.Tensor, 
                          bbox_center: torch.Tensor,
                          img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the bbox image as the model input by affine transform (Tensor version).

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (torch.Tensor): The bbox scale of the img.
        bbox_center (torch.Tensor): The bbox center of the img.
        img (torch.Tensor): The original image. Shape: (C, H, W) or (H, W, C)

    Returns:
        tuple: A tuple containing center and scale.
        - torch.Tensor: img after affine transform. Shape same as input.
        - torch.Tensor: bbox scale after affine transform.
    """
    # Ensure img is in channel-first format (C, H, W)
    # if img.ndim == 3 and img.shape[0] != 3:  # Assume (H, W, C) format
    #     img = img.permute(2, 0, 1)  # Convert to (C, H, W)
    
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    aspect_ratio = w / h
    b_w, b_h = bbox_scale[0, 0], bbox_scale[0, 1]
    
    if b_w > b_h * aspect_ratio:
        bbox_scale_new = torch.tensor([b_w, b_w / aspect_ratio], 
                                     dtype=bbox_scale.dtype, device=bbox_scale.device)
    else:
        bbox_scale_new = torch.tensor([b_h * aspect_ratio, b_h], 
                                     dtype=bbox_scale.dtype, device=bbox_scale.device)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale_new
    rot = 0
    warp_mat = get_warp_matrix(center.cpu().numpy(), scale.cpu().numpy(), rot, output_size=(w, h))  # For consistency check
    
    img_cv = img.cpu().numpy()  # Convert to (H, W, C) for cv2
    img_warped = cv2.warpAffine(img_cv, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    img_warped = torch.from_numpy(img_warped).to(dtype=img.dtype, device=img.device)  # Back to (C, H, W)

    # warpAffine using torch has some issues, so we use cv2 here
    # Prepare for affine transformation
    # warp_torch = torch.from_numpy(warp_mat).to(device=img.device, dtype = torch.float32)

    # C, H, W = img.shape
    
    # # Add batch dimension and convert to homogeneous coordinates
    # img_batch = img.unsqueeze(0).to(dtype=torch.float32)  # (1, C, H, W)
    
    # # Create grid and apply affine transformation
    # grid = F.affine_grid(warp_torch.unsqueeze(0), (1, C, h, w))
    # img_warped = F.grid_sample(img_batch, grid=grid, mode='bilinear', padding_mode='zeros')
    
    # # Remove batch dimension
    # img_warped = img_warped.squeeze(0)  # (C, h, w)

    # img_cv = img_warped.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    # cv2.imwrite('debug_original.jpg', img_cv)
    
    return img_warped, bbox_scale_new

def get_simcc_maximum_torch(simcc_x: torch.Tensor,
                      simcc_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (torch.Tensor): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (torch.Tensor): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (torch.Tensor): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (torch.Tensor): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = torch.argmax(simcc_x, dim=1)
    y_locs = torch.argmax(simcc_y, dim=1)
    locs = torch.stack((x_locs, y_locs), dim=-1).float()
    max_val_x = torch.amax(simcc_x, dim=1)
    max_val_y = torch.amax(simcc_y, dim=1)

    # get maximum value across x and y axis
    # mask = max_val_x > max_val_y
    # max_val_x[mask] = max_val_y[mask]
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

class RTMPose(BaseTool):

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (288, 384),
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        super().__init__(onnx_model, model_input_size, mean, std, backend, device)
        self.to_openpose = to_openpose

    def __call__(self, image: torch.Tensor, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [torch.tensor([0, 0, image.shape[2], image.shape[1]])]

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        keypoints, scores = [], []
        for bbox in bboxes:
            if bbox.ndim == 1:
                bbox = bbox.unsqueeze(0)
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = torch.concat(keypoints, dim=0)
        scores = torch.concat(scores, dim=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: torch.Tensor, bbox: torch.Tensor):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """

        # get center and scale
        center, scale = bbox_xyxy2cs_torch(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine_tensor_torch(self.model_input_size, scale, center, img)
        # normalize image
        if self.mean is not None:
            resized_img = resized_img.float()
            if not isinstance(self.mean, torch.Tensor):
                self.mean = torch.Tensor(self.mean).to(resized_img.device)
                self.std = torch.Tensor(self.std).to(resized_img.device)
            resized_img = (resized_img - self.mean) / self.std

        resized_img = resized_img.unsqueeze(0).permute(0, 3, 1, 2)  # B C H W

        return resized_img, center.to(device=resized_img.device), scale.to(device=resized_img.device)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        center: Tuple[int, int],
        scale: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum_torch(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        if isinstance(keypoints, np.ndarray):
            keypoints = keypoints / self.model_input_size * scale
            keypoints = keypoints + center - scale / 2
        elif isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.float()
            model_input_size_tensor = torch.tensor(self.model_input_size, dtype=keypoints.dtype, device=keypoints.device)

            keypoints = keypoints / model_input_size_tensor * scale
            keypoints = keypoints + center - scale / 2

        return keypoints, scores
