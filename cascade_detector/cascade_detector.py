import numpy as np
import torch
from .backend import OpenCVBackend, PillowBackend


class CascadeDetector:
    def __init__(self,
                 pnet,
                 rnet,
                 onet,
                 input_size=[12, 24, 48],
                 image_pyramid_factor=0.707,
                 min_face_size=15,
                 min_detection_stride=2,
                 normalize_mean=127.5,
                 normalize_std=128,
                 cls_thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7],
                 pnet_params=None,
                 rnet_params=None,
                 onet_params=None,
                 map_location=None,
                 device="cpu",
                 backend="pillow"):
        self._input_size = input_size
        self._image_pyramid_factor = image_pyramid_factor
        self._min_face_size = min_face_size
        self._min_detection_size = input_size[0]
        self._min_detection_stride = min_detection_stride
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self.cls_thresholds = cls_thresholds
        self.nms_thresholds = nms_thresholds
        self.device = device

        self.backend = None
        if backend.lower() in ("pillow", "pil"):
            self.backend = PillowBackend
        elif backend.lower() in ("opencv", "cv2"):
            self.backend = OpenCVBackend
        else:
            raise NotImplementedError

        # TODO: double check about multi-device compatible
        self.pnet = pnet.to(device)
        self.rnet = rnet.to(device)
        self.onet = onet.to(device)
        self.load_params(pnet_params, rnet_params, onet_params, map_location)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

    def __call__(self, image):
        return self.inference(image)

    def __repr__(self):
        repr_dict = {
            "image_pyramid":
            f"Image Pyramid(\n  min_face_size={self._min_face_size}\
                \n  min_detection_size={self._min_detection_size}\
                \n  factor:{self._image_pyramid_factor}\n)",
            "pnet":
            repr(self.pnet),
            "stage_bridge1":
            "Stage Bridge(\n" +
            f"  Choose(threshold={self.cls_thresholds[0]})\n" +
            f"  NMS(threshold={self.nms_thresholds[0]})\n" +
            f"  Resize(size={self._input_size[1]})\n)",
            "rnet":
            repr(self.rnet),
            "stage_bridge2":
            "Stage Bridge(\n" +
            f"  Choose(threshold={self.cls_thresholds[1]})\n" +
            f"  NMS(threshold={self.nms_thresholds[1]})\n" +
            f"  Resize(size={self._input_size[2]})\n)",
            "onet":
            repr(self.onet),
            "final_stage":
            "Output(\n" + f"  Choose(threshold={self.cls_thresholds[2]})\n" +
            f"  NMS(threshold={self.nms_thresholds[2]})\n)"
        }
        pipeline = ("image_pyramid", "pnet", "stage_bridge1", "rnet",
                    "stage_bridge2", "onet", "final_stage")
        repr_str = "MTCNN:"
        for key in pipeline:
            repr_str += f"\n\n{repr_dict[key]}"
        return repr_str

    @torch.no_grad()
    def inference(self, image):
        scale_pyramid = self.image_pyramid(image, self._min_detection_size,
                                           self._min_face_size,
                                           self._image_pyramid_factor)
        bboxes, offsets = self.pnet_stage(image, scale_pyramid,
                                          self.cls_thresholds[0],
                                          self._input_size[0],
                                          self._min_detection_stride)
        batch_image, bboxes = self.stage_bridge(image, bboxes, offsets,
                                                self.nms_thresholds[0],
                                                self._input_size[1])
        bboxes, offsets = self.rnet_stage(batch_image, bboxes,
                                          self.cls_thresholds[1])
        batch_image, bboxes = self.stage_bridge(image, bboxes, offsets,
                                                self.nms_thresholds[1],
                                                self._input_size[2])
        bboxes, offsets, landmarks = self.onet_stage(batch_image, bboxes,
                                                     self.cls_thresholds[2])
        bboxes, landmarks = self.final_stage(bboxes, offsets, landmarks,
                                             self.nms_thresholds[2])
        return bboxes.numpy(), landmarks.numpy()

    def image_pyramid(self,
                      image,
                      min_detection_size=12,
                      min_face_size=15,
                      factor=0.707):
        width, height = self.backend.get_image_size(image)
        min_length = min(width, height)
        m = min_detection_size / min_face_size
        min_length *= m
        scale_pyramid = []
        factor_power = 0
        while min_length > min_detection_size:
            scale = m * factor**factor_power
            scale_pyramid.append(scale)
            min_length *= factor
            factor_power += 1
        return scale_pyramid

    @torch.no_grad()
    def pnet_stage(self,
                   image,
                   scale_pyramid,
                   threshold,
                   cell_size=12,
                   stride=2):
        bboxes = []
        bboxes_offsets = []
        width, height = self.backend.get_image_size(image)
        for scale in scale_pyramid:
            scale_w = np.ceil(width * scale).astype(int)
            scale_h = np.ceil(height * scale).astype(int)
            resized_image = self.backend.resize(image, scale_w, scale_h)
            batch_data = self._preprocess(resized_image, self._normalize_mean,
                                          self._normalize_std)
            # TODO: double check about multi-device compatible
            batch_data = batch_data.to(self.device)
            probs, offsets = self.pnet(batch_data)
            probs = torch.softmax(probs, dim=1)[0, 1, :, :]

            indices = torch.where(probs > threshold)
            if indices[0].nelement() == 0:
                continue

            x1_offset, y1_offset, x2_offset, y2_offset = [
                offsets[0, i, indices[0], indices[1]] for i in range(4)
            ]
            scores = probs[indices[0], indices[1]]
            bbox = torch.stack(
                [
                    torch.round((stride * indices[1] + 1) / scale),
                    torch.round((stride * indices[0] + 1) / scale),
                    torch.round((stride * indices[1] + 1 + cell_size) / scale),
                    torch.round((stride * indices[0] + 1 + cell_size) / scale),
                    scores,
                ],
                dim=1,
            )
            offsets = torch.stack(
                [x1_offset, y1_offset, x2_offset, y2_offset],
                dim=1,
            )
            reserved = self.nms(bbox, 0.5)
            bboxes.append(bbox[reserved])
            bboxes_offsets.append(offsets[reserved])
        bboxes = torch.cat(bboxes)
        bboxes_offsets = torch.cat(bboxes_offsets)
        return bboxes, bboxes_offsets

    @torch.no_grad()
    def rnet_stage(self, batch_image, bboxes, threshold):
        # TODO: double check about multi-device compatible
        batch_image = batch_image.to(self.device)
        probs, offsets = self.rnet(batch_image)
        probs = torch.softmax(probs, dim=1)[:, 1]
        indices = torch.where(probs > threshold)[0]
        bboxes = bboxes[indices]
        bboxes[:, 4] = probs[indices]
        offsets = offsets[indices]
        return bboxes, offsets

    @torch.no_grad()
    def onet_stage(self, batch_image, bboxes, threshold):
        # TODO: double check about multi-device compatible
        batch_image = batch_image.to(self.device)
        probs, offsets, landmarks = self.onet(batch_image)
        probs = torch.softmax(probs, dim=1)[:, 1]
        indices = torch.where(probs > threshold)[0]
        bboxes = bboxes[indices]
        bboxes[:, 4] = probs[indices]
        offsets = offsets[indices]
        landmarks = landmarks[indices]
        return bboxes, offsets, landmarks

    def stage_bridge(self, image, bboxes, offsets, nms_threshold, resize):
        width, height = self.backend.get_image_size(image)
        bboxes = self._calibrate_boxes(bboxes, offsets)
        reserved = self.nms(bboxes, nms_threshold)
        bboxes = bboxes[reserved]

        bboxes = self._convert_to_square(bboxes)

        bboxes[:, 0:4] = torch.round(bboxes[:, 0:4])

        bboxes, coords, sizes = self._correct_boxes(bboxes, width, height)

        num_image = bboxes.size(0)
        batch_image = torch.zeros((num_image, 3, resize, resize),
                                  dtype=torch.float32)
        image = self.backend.to_array(image)
        for idx, (bbox, coord, size) in enumerate(zip(bboxes, coords, sizes)):
            cropped = self._crop_image_to_square(image, bbox, coord, size)
            resized = self.backend.resize(cropped, resize, resize)
            batch_image[idx] = self._preprocess(resized, self._normalize_mean,
                                                self._normalize_std)
        return batch_image, bboxes

    def final_stage(self, bboxes, offsets, landmarks, nms_threshold):
        x1, y1, x2, y2 = [bboxes[:, i].unsqueeze(1) for i in range(4)]
        bbox_w = x2 - x1 + 1
        bbox_h = y2 - y1 + 1
        landmarks[:, 0:5] = x1 + landmarks[:, 0:5] * bbox_w
        landmarks[:, 5:10] = y1 + landmarks[:, 5:10] * bbox_h

        bboxes = self._calibrate_boxes(bboxes, offsets)
        reserved = self.nms(bboxes, nms_threshold)
        bboxes = bboxes[reserved]
        landmarks = landmarks[reserved]
        return bboxes, landmarks

    def nms(self, bboxes, threshold):
        if len(bboxes) == 0:
            return []
        reserved = []
        x1, y1, x2, y2, score = [bboxes[:, i] for i in range(5)]
        sorted_score_indices = torch.argsort(score, descending=True)
        while len(sorted_score_indices) > 0:
            i = sorted_score_indices[0].item()
            reserved.append(i)
            iou = self._iou(
                (x1[i], y1[i], x2[i], y2[i]),
                (x1[sorted_score_indices], y1[sorted_score_indices],
                 x2[sorted_score_indices], y2[sorted_score_indices]))
            indices = torch.where(iou <= threshold)
            sorted_score_indices = sorted_score_indices[indices]
        return reserved

    def save_params(self,
                    save_pnet=True,
                    save_rnet=True,
                    save_onet=True,
                    filename="mtcnn"):
        if save_pnet:
            torch.save(self.pnet.state_dict(), filename + "-pnet.pt")
        if save_rnet:
            torch.save(self.rnet.state_dict(), filename + "-rnet.pt")
        if save_onet:
            torch.save(self.onet.state_dict(), filename + "-onet.pt")

    def load_params(self,
                    pnet_params=None,
                    rnet_params=None,
                    onet_params=None,
                    map_location=None):
        if pnet_params:
            self.pnet.load_state_dict(
                torch.load(pnet_params, map_location=map_location))
        if rnet_params:
            self.rnet.load_state_dict(
                torch.load(rnet_params, map_location=map_location))
        if onet_params:
            self.onet.load_state_dict(
                torch.load(onet_params, map_location=map_location))

    def _preprocess(self, image, mean=127.5, std=128):
        image = self.backend.to_rgb_array(image)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        image = (image - mean) / std
        return torch.from_numpy(image)

    def _iou(self, bbox, bboxes):
        inter_x1 = torch.max(bbox[0], bboxes[0])
        inter_y1 = torch.max(bbox[1], bboxes[1])
        inter_x2 = torch.min(bbox[2], bboxes[2])
        inter_y2 = torch.min(bbox[3], bboxes[3])
        inter_w = torch.threshold(inter_x2 - inter_x1 + 1, 0, 0)
        inter_h = torch.threshold(inter_y2 - inter_y1 + 1, 0, 0)
        inter_area = inter_w * inter_h
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        bboxes_area = (bboxes[2] - bboxes[0] + 1) * (bboxes[3] - bboxes[1] + 1)
        union_area = bbox_area + bboxes_area - inter_area
        return inter_area / union_area

    def _calibrate_boxes(self, bboxes, offsets):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        shift = torch.stack([w, h, w, h], dim=1) * offsets
        bboxes[:, 0:4] += shift
        return bboxes

    def _convert_to_square(self, bboxes):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        max_side = torch.max(w, h)
        bboxes[:, 0] = x1 + 0.5 * w - 0.5 * max_side
        bboxes[:, 1] = y1 + 0.5 * h - 0.5 * max_side
        bboxes[:, 2] = bboxes[:, 0] + max_side - 1
        bboxes[:, 3] = bboxes[:, 1] + max_side - 1
        return bboxes

    def _correct_boxes(self, bboxes, width, height):
        coords = torch.zeros_like(bboxes[:, 0:4])

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        bboxes_w = x2 - x1 + 1
        bboxes_h = y2 - y1 + 1
        sizes = torch.stack((bboxes_w, bboxes_h), dim=1)
        coords[:, 2] = bboxes_w - 1
        coords[:, 3] = bboxes_h - 1

        indices = torch.where(x2 > width - 1)[0]
        coords[:, 2][indices] = width - x1[indices] - 1
        bboxes[:, 2][indices] = width - 1

        indices = torch.where(y2 > height - 1)[0]
        coords[:, 3][indices] = height - y1[indices] - 1
        bboxes[:, 3][indices] = height - 1

        indices = torch.where(x1 < 0)[0]
        coords[:, 0][indices] = -x1[indices]
        bboxes[:, 0][indices] = 0

        indices = torch.where(y1 < 0)[0]
        coords[:, 1][indices] = -y1[indices]
        bboxes[:, 1][indices] = 0

        return bboxes, coords, sizes

    def _crop_image_to_square(self, image, bbox, coord, size):
        x1, y1, x2, y2 = [bbox[i].to(torch.int).item() for i in range(4)]
        crop_x1, crop_y1, crop_x2, crop_y2 = [
            coord[i].to(torch.int).item() for i in range(4)
        ]
        crop_w, crop_h = [size[i].to(torch.int).item() for i in range(2)]

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        cropped[crop_y1:crop_y2 + 1,
                crop_x1:crop_x2 + 1, :] = image[y1:(y2 + 1), x1:(x2 + 1), :]
        cropped = self.backend.from_array(cropped)
        return cropped
