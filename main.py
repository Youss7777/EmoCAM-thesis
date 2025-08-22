import analysis.local_analysis as la
import analysis.global_analysis as ga
from my_config import Config
from explanations.emonet import ExplanationsEmonet


def main():
    cfg = Config()
    expl_emonet = ExplanationsEmonet(config=cfg)
    max_class, max_prob, pred, arousal, valence, vis, grayscales = expl_emonet.get_explanations_for_image(img_path=cfg.PATH_IMAGE,
                                                                                                          show_plot=True)
    la.LocalAnalysis().local_analysis(file_path=cfg.PATH_IMAGE, max_emotion=max_class, max_prob=max_prob, cam_output=grayscales,
                                           nb_objects=5, confidence_thresh=0.3, show_output=True)
    ga.GlobalAnalysis(config=cfg, device=cfg.device, nb_objects=5, confidence_tresh=0.3).analyze()


if __name__ == '__main__':
    main()
