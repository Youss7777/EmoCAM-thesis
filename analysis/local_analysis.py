"""
Apply YOLOv3 and explanation methods on a single image
"""
import os

import brambox as bb
import lightnet as ln
import numpy as np
import torch
from PIL import Image
from lightnet.models import YoloV3
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from my_config import Config

from preprocess.img_resize import ImgResize
from pytorch_grad_cam.utils.image import show_cam_on_image


class LocalAnalysis:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        cfg = Config()
        # Load names of OpenImage classes
        class_map = []
        with open(cfg.PATH_OPENIMAGES_CLASSES, "r") as fin:
            for line in fin:
                line = line.strip()
                class_map.append(line)

        # Load YoloV3 model + OpenImage weights
        self.model = YoloV3(601)
        self.model.load(Config.PATH_YOLOV3_WEIGHTS)
        self.model.eval()
        self.model.to(device)

        thresh = 0.005

        # Create post-processing pipeline
        self.post = ln.data.transform.Compose([
            # GetBoxes transformation generates bounding boxes from network output
            ln.data.transform.GetMultiScaleAnchorBoxes(
                conf_thresh=thresh,
                network_stride=self.model.stride,
                anchors=self.model.anchors
            ),

            # Filter transformation to filter the output boxes
            ln.data.transform.NMS(
                iou_thresh=thresh
            ),

            # Miscelaneous transformation that transforms the output boxes to a brambox dataframe
            ln.data.transform.TensorToBrambox(
                class_label_map=class_map,
            )
        ])

        img_resize = ImgResize(width=608, height=608)
        self.transform = transforms.Compose([img_resize])

    @staticmethod
    def _confidence_cutoff(df, threshold):
        df['importance'] = df['object_importance']
        df.loc[df['confidence'] < threshold, 'importance'] = 0
        return df

    @staticmethod
    def _add_importance(df, heatmap):
        importance = []
        for index, row in df.iterrows():
            x_min = int(row["x_top_left"])
            x_max = int(row["x_top_left"] + row["width"])
            y_min = int(row["y_top_left"])
            y_max = int(row["y_top_left"] + row["height"])
            # region inside the bounding box
            bounded_region = heatmap[y_min:y_max, x_min:x_max]
            # define importance as the average activation inside that region
            importance.append(np.mean(bounded_region))
        df["object_importance"] = importance
        return df

    def local_analysis(self, file_path, max_emotion, max_prob, nb_objects,
                       cam_output, confidence_thresh, show_output=False):
        """
        Perform local analysis on single image.
        """
        img_path = os.path.join(file_path)
        # load image
        with torch.no_grad():
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img)
            # resize heatmap to input image
            original_size_img = img.size
            # for one method
            method_outputs = []
            for grayscale in cam_output:
                grayscale_cam = cv2.resize(grayscale, original_size_img)
                grayscale_cam_pil = Image.fromarray(grayscale_cam)
                grayscale_cam_tensor = self.transform(grayscale_cam_pil)
                grayscale_cam_scaled = grayscale_cam_tensor.numpy()[0, :]
                # get yolo output
                output_tensor = self.model(img_tensor.unsqueeze(0).to(self.device))
                # post-processing
                output_df = self.post(output_tensor)
                proc_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
                # superimpose image and gradcam heatmap
                cam = show_cam_on_image(proc_img, grayscale_cam_scaled, use_rgb=True)
                pil_img = Image.fromarray(cam)
                # add importance of bounding boxes
                df_complete = self._add_importance(output_df, grayscale_cam_scaled)
                # importance confidence cutoff
                df_complete = self._confidence_cutoff(df_complete, threshold=confidence_thresh)
                # rename 'class_label' to 'detected_object' for more clarity later
                df_complete_return = df_complete.rename(columns={'class_label': 'detected_object',
                                                                 'confidence': 'object_confidence'}).sort_values(by="importance", ascending=False)
                df_sorted = df_complete.sort_values(by="importance", ascending=False)
                method_outputs.append((pil_img, df_sorted))

            if show_output:
                fig, ax = plt.subplots(1, len(method_outputs))
                if len(method_outputs) == 1:
                    ax = [ax]
                for i, (pil_img, df_sorted) in enumerate(method_outputs):
                    df_sorted = df_sorted.head(nb_objects)
                    print(df_sorted)
                    ax[i].imshow(bb.util.draw_boxes(pil_img, df_sorted, label=df_sorted.class_label))
                    ax[i].axis('off')
                    object_list = '\n'.join([f'{obj}: {imp*100:0.1f}%' for obj, imp in zip(df_sorted['class_label'],
                                                                                           df_sorted['object_importance'])])
                    fig.text(0.5, 0.97, f'{max_emotion}: {max_prob*100:.1f}%'+'\n'+object_list, ha='center', va='top',
                             fontsize=13, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                    plt.subplots_adjust(top=0.83)
                    plt.show()

        return df_complete_return
