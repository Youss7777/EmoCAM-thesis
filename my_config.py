import os
import torch
from explanations.visualizations import Processors


class Config:
    PATH_YOLOV3_WEIGHTS = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM',
                                       'weight', 'yolov3-openimages.weights')
    PATH_IMAGE = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                              'test_images', 'kids_smiling.jpg')
    PATH_IMG_MEAN = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                                 'img_mean.txt')
    PATH_EMONET = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                               'emonet.pth')
    PATH_OPENIMAGES_CLASSES = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                                           'openimages.names')
    PATH_DATASET = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'PytorchProject',
                                'emonet-master', 'emonet_py', 'findingemo_dataset')
    PATH_ANNOTATIONS = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                                    'annotations_single.ann')
    PATH_MODEL_OUTPUTS = os.path.join(os.path.expanduser('~'), 'Documents', 'AI Master', 'Thesis', 'EmoCAM', 'data',
                                      'outputs')
    # Choose one of several explanation method(s)
    METHODS = [Processors.GRADCAM]

    def __init__(self, device=torch.device('cpu')):
        self.device = device

