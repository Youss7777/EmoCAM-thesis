from models.emonet import EmoNet, EmoNetPreProcess
from models.emonet_arousal import EmoNetArousal
from models.emonet_valence import EmoNetValence
from explanations.visualizations import FileProcessor
import torch
import numpy as np
from my_config import Config


class ExplanationsEmonet:
    header = ['img_path', 'emonet_adoration_prob', 'emonet_aesthetic_appreciation_prob', 'emonet_amusement_prob',
              'emonet_anxiety_prob', 'emonet_awe_prob', 'emonet_boredom_prob', 'emonet_confusion_prob',
              'emonet_craving_prob', 'emonet_disgust_prob', 'emonet_empathetic_pain_prob', 'emonet_entrancement_prob',
              'emonet_excitement_prob', 'emonet_fear_prob', 'emonet_horror_prob', 'emonet_interest_prob',
              'emonet_joy_prob', 'emonet_romance_prob', 'emonet_sadness_prob', 'emonet_sexual_desire_prob',
              'emonet_surprise_prob', 'emonet_valence', 'emonet_arousal', 'annotation_user',
              'annotation_original_img_path', 'annotation_reject', 'annotation_tag', 'annotation_age_group',
              'annotation_valence', 'annotation_arousal', 'annotation_emotion', 'annotation_deciding_factors',
              'annotation_ambiguity', 'annotation_fmri_candidate', 'annotation_datetime']

    def __init__(self, config: Config,  device=torch.device('cpu')):
        self.device = device
        self.emonet = EmoNet(path_emonet=config.PATH_EMONET).get_emonet()
        self.emonet.to(device)
        self.emonet_pp = EmoNetPreProcess(path_img_mean=config.PATH_IMG_MEAN)
        self.emonet_arousal = EmoNetArousal(emonet=self.emonet).to(device)
        self.emonet_valence = EmoNetValence(emonet=self.emonet).to(device)
        self.file_processor = FileProcessor(model=self.emonet, target_layers=self.emonet.get_target_layers(),
                                            methods=config.METHODS)

    def get_most_probable_class(self, preds: torch.Tensor):
        max_prob = preds[0][0]
        max_class = EmoNet.EMOTIONS[0]
        class_index = 0
        for sample in range(preds.shape[0]):
            for emo_idx in range(20):
                if preds[sample, emo_idx] > max_prob:
                    max_prob = preds[sample, emo_idx]
                    max_class = EmoNet.EMOTIONS[emo_idx]
                    class_index = emo_idx
        return max_prob.item(), max_class, class_index

    def get_explanations_for_image(self, img_path, show_plot=False):
        # Instantiations & definitions
        img_tensor, img_loaded = self.emonet_pp(img_path)
        img_size = img_loaded.size
        in_tensor = img_tensor.unsqueeze(0)

        # We don't need the gradients
        with torch.no_grad():
            in_tensor = in_tensor.to(self.device)
            arousal = self.emonet_arousal(in_tensor).item()
            valence = self.emonet_valence(in_tensor).item()
            pred = self.emonet(in_tensor)

        # Processed image
        proc_img = np.float32(img_loaded)/255
        # Model output
        max_prob, max_class, class_index = self.get_most_probable_class(pred)
        # Visualization
        vis, grayscales = self.file_processor.get_visualizations(image=proc_img, input_tensor=in_tensor,
                                                                 class_index=class_index, img_size=img_size)
        if show_plot:
            self.file_processor.plot_cam(visualization=vis, image=img_loaded, class_label=max_class, prob_label=max_prob,
                                         val=valence, aro=arousal)

        return max_class, max_prob, pred, arousal, valence, vis, grayscales

