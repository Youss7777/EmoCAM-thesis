from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, EigenCAM, guided_backprop
from pytorch_grad_cam.utils.image import show_cam_on_image

from explanations.methods import CAM_Explanation


class Processors:
    # BEWARE! CONSTANT FIELDS ARE DEFINED AT THE BOTTOM, SO AS TO BE ABLE TO REFERENCE THESE FOLLOWING METHODS
    # They need to be defined BEFORE referencing them in the static fields.
    @staticmethod
    def process_type_a(processor, input_tensor, targets, image, image_weight, **kwargs):
        """
        Type of processing method required for a specific class of CAM methods (e.g., GradCAM, GradCAM++, AblationCAM,
        ScoreCAM, EigenCAM)

        :param processor:
        :param input_tensor:
        :param targets:
        :param image:
        :param image_weight:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor=input_tensor, targets=targets)
        grayscale = grayscale[0, :]
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=image_weight)

        return grayscale, vis

    @staticmethod
    def process_type_b(processor, input_tensor, class_index, img_size, image, image_weight, **kwargs):
        """
        Type of processing method required for a specific class of CAM methods (e.g., LiftCAM, LRPCAM, LimeCAM)
        :param processor:
        :param input_tensor:
        :param class_index:
        :param img_size:
        :param image:
        :param image_weight:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor, int(class_index), img_size)
        grayscale = torch.squeeze(grayscale).cpu().detach().numpy()
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=image_weight)

        return grayscale, vis

    @staticmethod
    def process_type_c(processor, input_tensor, class_index, targets, image, **kwargs):
        """
        Type of processing method required for a specific class of CAM methods (e.g., GuidedBackProp)
        :param processor:
        :param input_tensor:
        :param class_index:
        :param targets:
        :param image:
        :param kwargs: stuff we can ignore, but this way we can send the same parameters to all process_type methods
        :return:
        """
        grayscale = processor(input_tensor=input_tensor, targets=targets)
        grayscale = grayscale[0, :]
        vis = show_cam_on_image(image, grayscale, use_rgb=True, image_weight=0.0)
        # Get guided backpropagation map
        camGuided = guided_backprop.GuidedBackpropReLUModel(processor.model, "cpu")
        grayscale_cam_Guided = camGuided(input_tensor, class_index)
        # Elementwise multiplication
        guidedmap = grayscale_cam_Guided * vis

        return grayscale, (("Guided Backprop", grayscale_cam_Guided), ("Guided Grad-Cam", guidedmap))

    # Define static fields
    GRADCAM = (GradCAM, process_type_a, 'GradCAM')
    GUIDED = (GradCAM, process_type_c, 'Guided')
    GRADCAMPP = (GradCAMPlusPlus, process_type_a, 'GradCAM++')
    ABLATIONCAM = (AblationCAM, process_type_a, 'AblationCAM')
    SCORECAM = (ScoreCAM, process_type_a, 'ScoreCAM')
    EIGENCAM = (EigenCAM, process_type_a, 'EigenCAM')
    LIFTCAM = ((CAM_Explanation, {'method': 'LIFT-CAM'}), process_type_b)
    LRPCAM = ((CAM_Explanation, {'method': 'LRP-CAM'}), process_type_b)
    LIMECAM = ((CAM_Explanation, {'method': 'LIME-CAM'}), process_type_b)


class FileProcessor:
    def __init__(self, model, target_layers, methods: Iterable[Processors] = None):
        self.processors = {}
        for method in methods:
            if isinstance(method[0], tuple):
                # method[0] = (CAM method, parameters), method[1] = process method to be used with CAM method
                proc_name, proc_params = str(method[0][0]), method[0][1]
                proc_params['model'] = model
                processor = method[0][0](**proc_params)
                self.processors[proc_name] = (processor, method[1])
            else:
                # method[0] = CAM method, method[1] = process method to be used with CAM method, method[2] = name
                proc_name = method[2]
                print(f'processor : {proc_name}')
                print(f'processor_method : {method[0]}')
                processor = method[0](model=model, target_layers=target_layers)
                self.processors[proc_name] = (processor, method[1])
                #     example for GradCAM only:
                #     methods = [(GradCAM, process_type_a, 'GradCAM')]
                #     processors = {'GradCAM': (GradCAM(model, target_layers), process_type_a)}

    def get_visualizations(self, image: np.ndarray,
                           input_tensor: torch.Tensor,
                           class_index: int,
                           img_size,
                           image_weight: float = 0.5,
                           targets=None):
        """
        Get visualizations for a single image

        :param image: numpy array representation of the image to process
        :param input_tensor:
        :param class_index:
        :param img_size:
        :param image_weight:
        :param targets:
        :return: vis, grayscales
        """
        vis = []
        grayscales = []
        # Keep track of CAM output we are generating, then delete afterwards
        # saved_files = set()
        for method, processor_info in self.processors.items():
            processor, processor_method = processor_info
            # example for gradCAM only
            # method = GradCAM(model, target_layers)
            # processor_info = process_type_a
            params = {'processor': processor, 'input_tensor': input_tensor,
                      'targets': targets, 'image': image, 'image_weight': image_weight,
                      'class_index': class_index, 'img_size': img_size}
            grayscale, vis_cam = processor_method(**params)
            grayscales.append(grayscale)
            if len(vis_cam) == 2:  # type_c returns 2, a and b 1 CAM result(s)
                for tpl in vis_cam:
                    vis.append([tpl[0], tpl[1]])
            else:
                vis.append([method, vis_cam])

        return vis, grayscales

    @staticmethod
    def plot_cam(visualization, image, class_label, prob_label, val, aro):
        """
        plot the different localization maps superimposed on image with the most probable class, valence and arousal
        predicted by EmoNet
        """
        ncol = len(visualization)+1
        fig, ax = plt.subplots(1, ncol)
        ax[0].imshow(image, interpolation='none')
        ax[0].axis('off')
        ax[0].set_title("Image")
        for i in range(len(visualization)):
            ax[i+1].imshow(visualization[i][1], interpolation='none')
            ax[i+1].axis('off')
            ax[i+1].set_title(visualization[i][0])
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.suptitle(f"Class: {class_label}\nConfidence: {prob_label*100:0.2f}% \nValence: {val:.0f} \nArousal: {aro:.0f}")
        plt.show()
