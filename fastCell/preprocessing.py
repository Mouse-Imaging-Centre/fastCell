import argparse

def add_preprocessing_group(master_parser: argparse.ArgumentParser) -> None:
    preprocessing_group = master_parser.add_argument_group("Preprocessing")
    preprocessing_group.add_argument("--crop-edges", dest="crop_edges", action="store_true", default=True,
                       help = "Crop the edges if the image is not divisible into 224*224 tiles.")