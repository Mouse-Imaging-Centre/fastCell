import argparse

def add_segmentation_group(master_parser: argparse.ArgumentParser) -> None:
    segmentation_group = master_parser.add_argument_group("Segmentation")
    segmentation_group.add_argument("--learner", dest="learner", type=str, required=True,
                                     help = "Load the Learner object that was saved from export().")
    segmentation_group.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=False,
                                     help = "Load the Learner object on the gpu instead of the cpu.")
    segmentation_group.add_argument("--fp32", dest="fp32", action="store_true", default=True,
                                     help = "Put the learner in FP32 precision mode. "
                                            "You may need this if your weights are half-precision.")