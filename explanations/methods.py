"""
Implementations of LIFT‑CAM, LRP‑CAM and LIME‑CAM.

This module contains utility functions and a callable class
(:class:`CAM_Explanation`) used to compute visual explanations for
convolutional neural networks.  The explanations are based on
attribution methods provided by Captum (DeepLift, LRP and Lime) and
produce class activation maps (CAMs) which can be overlaid on input
images to highlight relevant regions.

The functions defined here operate on a variety of supported models
including AlexNet variants, VGG and ResNet.  See the accompanying
scripts in the ``analysis`` package for example usage.
"""

import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
from utils.lift_utils import min_max_normalize
from captum.attr import DeepLift, LRP, Lime
from models import alexnet_big


# The implementations of LIFT-CAM, LRP-CAM, and LIME-CAM
class CAM_Explanation:
    def __init__(self, model, method):
        self.model = model
        self.method = method

    def __call__(self, x, class_id, image_size):
        if torch.cuda.is_available():
            x = x.cuda()
        if self.method == "LIFT-CAM":
            explanation = lift_cam(self.model, x, class_id)
        elif self.method == "LRP-CAM":
            explanation = lrp_cam(self.model, x, class_id)
        elif self.method == "LIME-CAM":
            explanation = lime_cam(self.model, x, class_id)
        else:
            raise Exception("Not supported method.")

        with torch.no_grad():
            explanation = F.interpolate(explanation, size=image_size[::-1], mode="bilinear")
        explanation = explanation.detach().cpu()
        explanation = min_max_normalize(explanation)
        return explanation


# The later part of a given original prediction model
class ModelPart(nn.Module):
    def __init__(self, model):
        super(ModelPart, self).__init__()
        if isinstance(model, torchvision.models.vgg.VGG):
            self.model_type = "vgg16"
            self.max_pool = model.features[-1:]
            self.avg_pool = model.avgpool
            self.classifier = model.classifier
        elif isinstance(model, torchvision.models.resnet.ResNet):
            self.model_type = "resnet50"
            self.avg_pool = model.avgpool
            self.classifier = model.fc
        elif isinstance(model, alexnet_big.AlexNetBigEmoNet):
            self.model_type = "emonet"
            self.classifier = [model.fc1, model.fc2, model.fc3]
            self.max_pool = model.maxpool
        else:
            raise Exception("Model not supported.")

    def forward(self, x):
        if self.model_name == "emonet":
            x = self.max_pool(F.relu(x))
            x = x.permute((0, 1, 3, 2)).reshape(x.shape[0], -1)
            x = F.relu(self.classifier[0](x))
            x = F.relu(self.classifier[1](x))
            x = self.classifier[2](x)
        else:
            if self.model_type == "vgg16":
                x = self.max_pool(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


def get_handle_act_map(model, x, class_id):
    act_map_list = []

    def forward_hook(output):
        act_map_list.append(output)
    if isinstance(model, torchvision.models.vgg.VGG):
        handle = model.features[-2].register_forward_hook(forward_hook)
    elif isinstance(model, torchvision.models.resnet.ResNet):
        handle = model.layer4.register_forward_hook(forward_hook)
    elif isinstance(model, alexnet_big.AlexNetBigEmoNet):
        handle = model.conv5.register_forward_hook(forward_hook)
    else:
        raise Exception("Not supported architecture.")
    output = model(x)
    if class_id is None:
        class_id = torch.argmax(output, dim=1)
    act_map = act_map_list[0]
    handle.remove()
    del act_map_list

    return handle, act_map, class_id


def get_expl_map(contributions, act_map, method):
    if method == "lift" or "lrp":
        scores_temp = torch.sum(contributions, (2, 3), keepdim=False)
    elif method == "lime":
        scores_temp = contributions
    scores = torch.squeeze(scores_temp, 0)
    scores = scores.cpu()
    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()
    return vis_ex_map


# LIFT-CAM
def lift_cam(model, x, class_id=None):
    handle, act_map, class_id = get_handle_act_map(model, x, class_id)
    model_part = ModelPart(model)
    model_part.eval()
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map)
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False)
    vis_ex_map = get_expl_map(dl_contributions, act_map, method='lift')
    
    return vis_ex_map


# LRP-CAM
def lrp_cam(model, x, class_id=None):
    handle, act_map, class_id = get_handle_act_map(model, x, class_id)
    model_part = ModelPart(model)
    model_part.eval()
    lrp = LRP(model_part)
    lrp_contributions = lrp.attribute(act_map, target=class_id, return_convergence_delta=False)
    vis_ex_map = get_expl_map(lrp_contributions, act_map, method='lrp')
    return vis_ex_map


# LIME-CAM
def lime_cam(model, x, class_id=None):
    handle, act_map, class_id = get_handle_act_map(model, x, class_id)
    nb_channels = int(act_map.size(1))
    model_part = ModelPart(model)
    model_part.eval()
    lime = Lime(model_part)
    ref_map = torch.zeros_like(act_map)
    f_mask = torch.zeros_like(act_map).long()
    for i in range(f_mask.size(1)):
        f_mask[:, i, :, :] = i
    lime_contributions = lime.attribute(act_map, baselines=ref_map, n_samples=nb_channels,
                                        target=class_id, feature_mask=f_mask, return_input_shape=False)
    vis_ex_map = get_expl_map(lime_contributions, act_map, method='lime')
    return vis_ex_map
