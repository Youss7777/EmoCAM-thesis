"""
Local analysis using Detectron for object detection.

This module implements a pipeline similar to
``local_analysis.py`` but utilises a Detectron2 model for object
detection instead of YOLO.  Detected bounding boxes are combined
with Grad‑CAM heatmaps to compute the importance of each object in
eliciting a particular emotion.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detectron:

    def __init__(self, img):
        self.cfg = get_cfg()
        self.img = img
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(self.cfg)
        self.outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()


    def compare(self):
        heatmap = np.load("cam_grad_dataset/cam_grad_" + file_name + ".npy")  # Assuming heatmap is saved as a NumPy array
        # load image and heatmap
        image = self.img
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for consistency
        # resize heatmap
        resized_heatmap = resize_heatmap(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
        # overlay heatmap on image
        heatmap_overlay = overlay_heatmap_on_image(image_rgb, resized_heatmap)
        # filter polygons by importance
        filtered_indices = filter_polygons_by_importance(self.outputs['instances'], resized_heatmap, 0.1)
        # visualize result
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # visualize without filtering
        visualize_overlay(image, self.outputs['instances'], metadata, heatmap_overlay)
        # visualize with filtering
        visualize_filtered_predictions(image_rgb, self.outputs['instances'], filtered_indices, metadata, heatmap_overlay)


# resize the heatmap
def resize_heatmap(heatmap, target_size):
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

# overlay heatmap
def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert heatmap to RGB
    overlay = cv2.addWeighted(image, alpha, heatmap_colored, 1 - alpha, 0)
    return overlay

# filter polygons by importance
def filter_polygons_by_importance(instances, binary_mask, importance_threshold):
    filtered_indices = []
    for i, mask in enumerate(instances.pred_masks):
        mask = mask.numpy()
        importance = np.sum(binary_mask * mask) / np.sum(mask)
        if importance >= importance_threshold:
            filtered_indices.append(i)
    return filtered_indices


# visualize result
def visualize_filtered_predictions(image, instances, filtered_indices, metadata, heatmap_overlay):
    filtered_instances = instances[filtered_indices]
    v = Visualizer(image[:, :, ::-1], metadata=metadata)  # Convert image to RGB for Visualizer
    v = v.draw_instance_predictions(filtered_instances)
    result_image = v.get_image()[:, :, ::-1]  # Convert back to BGR for consistency with OpenCV
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.imshow(heatmap_overlay, alpha=0.5)
    plt.axis('off')
    plt.show()

def visualize_overlay(image, instances, metadata, heatmap_overlay):
    v = Visualizer(image[:, :, ::-1], metadata=metadata)  # Convert image to RGB for Visualizer
    v = v.draw_instance_predictions(instances)
    result_image = v.get_image()[:, :, ::-1]  # Convert back to BGR for consistency with OpenCV
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.imshow(heatmap_overlay, alpha=0.5)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    file_name = 'basketball_player1.jpg'
    file_path = 'test_images/' + file_name
    image = cv2.imread(file_path)
    detectron = Detectron(image)
    detectron.compare()

