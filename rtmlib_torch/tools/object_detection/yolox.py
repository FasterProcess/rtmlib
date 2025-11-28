# Code modified from https://github.com/IDEA-Research/DWPose/blob/opencv_onnx/ControlNet-v1-1-nightly/annotator/dwpose/cv_ox_det.py  # noqa
from typing import List, Tuple

import cv2
import numpy as np
import torch, torchvision
import torch.nn.functional as F

from ..base import BaseTool
from .post_processings import multiclass_nms


def multiclass_nms_torch(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in PyTorch.

    Class-aware version.
    """
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr

        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]

            # 使用 torchvision.ops.nms
            keep = torchvision.ops.nms(valid_boxes, valid_scores, nms_thr)

            if len(keep) > 0:
                cls_inds = torch.ones((len(keep), 1), device=boxes.device) * cls_ind
                dets = torch.cat(
                    [valid_boxes[keep], valid_scores[keep].unsqueeze(1), cls_inds],
                    dim=1,
                )
                final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return torch.cat(final_dets, 0)


class YOLOX(BaseTool):

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (640, 640),
        nms_thr=0.45,
        score_thr=0.7,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        super().__init__(onnx_model, model_input_size, backend=backend, device=device)
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image, ratio = self.preprocess(image)
        batch_size = image.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.inference(image[i : i + 1])[0]
            outputs.append(out)
        # outputs = self.inference(image)[0]
        results = self.postprocess(outputs, ratio)
        return results

    def preprocess(self, img: torch.Tensor):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        b, h, w, c = img.shape
        padded_img = (
            torch.ones(
                b,
                c,
                self.model_input_size[0],
                self.model_input_size[1],
                dtype=img.dtype,
                device=img.device,
            )
            * 114
        )

        ratio = min(self.model_input_size[0] / h, self.model_input_size[1] / w)
        img = img.permute(0, 3, 1, 2)  # B H W C -> B C H W
        resized_img = F.interpolate(
            img,
            size=(int(h * ratio), int(w * ratio)),
            mode="bilinear",
            align_corners=False,
        )
        padded_shape = (int(h * ratio), int(w * ratio))
        padded_img[..., : padded_shape[0], : padded_shape[1]] = resized_img

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[torch.Tensor],
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMPose model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMPose model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        # outputs = torch.cat(outputs, dim=0)

        boxes_result = []
        scores_result = []

        if len(outputs) < 1:
            return boxes_result, scores_result
        
        if isinstance(outputs[0], torch.Tensor) and not isinstance(self.nms_thr, torch.Tensor):
            self.nms_thr = torch.tensor(self.nms_thr, device=outputs[0].device)
            self.score_thr = torch.tensor(self.score_thr, device=outputs[0].device)      

        if outputs[0].shape[-1] == 85:
            for i in range(len(outputs)):
                output = outputs[i]
                device = output.device
                grids = []
                expanded_strides = []
                strides = [8, 16, 32]

                hsizes = [self.model_input_size[0] // stride for stride in strides]
                wsizes = [self.model_input_size[1] // stride for stride in strides]

                for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                    # 使用 torch.meshgrid 替代 np.meshgrid
                    # 注意：torch.meshgrid 默认使用 'ij' 索引，而 numpy 使用 'xy' 索引
                    # 所以我们需要交换顺序或使用 indexing='ij'
                    yv, xv = torch.meshgrid(
                        torch.arange(hsize, device=device),
                        torch.arange(wsize, device=device),
                        indexing="ij",
                    )

                    # 堆叠并重塑为 [1, H*W, 2]
                    grid = torch.stack((xv, yv), 2).reshape(1, -1, 2)
                    grids.append(grid)

                    # 创建与 grid 前两个维度相同的 expanded_strides
                    shape = grid.shape[:2]  # [1, H*W]
                    expanded_stride = torch.full(
                        (*shape, 1), stride, device=device
                    )
                    expanded_strides.append(expanded_stride)

                # 拼接 grids 和 expanded_strides
                grids = torch.cat(grids, 1)
                expanded_strides = torch.cat(expanded_strides, 1)

                # 解码边界框坐标和尺寸
                output[..., :2] = (output[..., :2] + grids) * expanded_strides
                output[..., 2:4] = torch.exp(output[..., 2:4]) * expanded_strides

                # 获取预测结果
                predictions = output[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4:5] * predictions[:, 5:]

                # 将边界框从 (中心x, 中心y, 宽, 高) 转换为 (x1, y1, x2, y2) 格式
                boxes_xyxy = torch.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
                boxes_xyxy /= ratio
                dets = multiclass_nms_torch(
                    boxes_xyxy, scores, nms_thr=self.nms_thr, score_thr=self.score_thr
                )

                if dets is not None:
                    pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                    final_boxes, final_scores, final_cls_inds = pack_dets
                    isscore = final_scores > 0.3
                    iscat = final_cls_inds == 0
                    isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                    final_boxes = final_boxes[isbbox] if isscore.sum() > 0 else None
                    final_scores = final_scores[isbbox] if isscore.sum() > 0 else None
                    boxes_result.append(final_boxes)
                    scores_result.append(final_scores)

        elif outputs[0].shape[-1] == 5:
            # onnx contains nms module
            for i in range(len(outputs)):
                pack_dets = (outputs[i][..., :4], outputs[i][..., 4])
                final_boxes, final_scores = pack_dets
                final_boxes /= ratio
                isscore = final_scores > 0.3
                # isscore[0] = False
                # isbbox = (i for i in isscore)
                final_boxes = final_boxes[isscore] if isscore.sum() > 0 else None
                final_scores = final_scores[isscore] if isscore.sum() > 0 else None

                boxes_result.append(final_boxes)
                scores_result.append(final_scores)

        return boxes_result, scores_result
