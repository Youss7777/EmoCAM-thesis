
import torch

from my_config import Config


class ModelTools:
    @staticmethod
    def get_most_probable_class(preds: torch.Tensor):
        """

        :param preds: 1D tensor containing the probabilities for each class
        :param class_labels: the corresponding class labels
        :return:
        """
        max_prob = preds[0]
        class_index = 0
        for emo_idx in range(len(preds)):
            if preds[emo_idx] > max_prob:
                max_prob = preds[emo_idx]
                class_index = emo_idx

        return max_prob.item(), class_index

    @staticmethod
    def load_openimages_classnames():
        """
        :return: list containing the Open Images classnames
        """
        class_map = []
        with open(Config.FILE_OPENIMAGES_CLASSES, "r") as fin:
            for line in fin:
                line = line.strip()
                class_map.append(line)

        return class_map
