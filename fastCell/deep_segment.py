import argparse
import tempfile
import fastai.vision
import torch
import cv2 as cv
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Segment the cells from an image.')
parser.add_argument("--images", dest="images", type=str, nargs="+", required=True,
                    help = "Images to segment")
parser.add_argument("--learner", dest="learner", type=str, required=True,
                   help = "Load the Learner object that was saved from export().")
parser.add_argument("--output-directory", dest="output_directory", type=str, required=True,)
parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=False,
                   help = "Load the Learner object on the gpu instead of the cpu.")
parser.add_argument("--crop-edges", dest="crop_edges", action="store_true", default=True,
                   help = "Crop the edges if the image is not divisible into 224*224 tiles.")
parser.add_argument("--temp-dir", dest="temp_dir", type=str, default=None)
parser.add_argument("--keep-temp", dest="keep_temp", action="store_true", default=False)

args = parser.parse_args()

l = 224

if __name__ == '__main__':
    output_dir = Path(args.output_directory)

    fastai.torch_core.defaults.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    #this has to be done because of how stupidly learner() takes its arguments
    learner_path = Path(args.learner)
    learn = fastai.basic_train.load_learner(path = learner_path.parent, file = learner_path.name)

    if args.keep_temp:
        temp_dir = Path(tempfile.mkdtemp(dir=args.temp_dir))
    else:
        temp_object = tempfile.TemporaryDirectory(dir=args.temp_dir)
        temp_dir = Path(temp_object.name)

    for image_path in [Path(image) for image in args.images]:
        image = cv.imread(image_path.as_posix(), cv.COLOR_BGR2GRAY)

        if args.crop_edges:
            image = image[image.shape[0] % l // 2:image.shape[0] - image.shape[0] % l // 2,
                    image.shape[1] % l // 2:image.shape[1] - image.shape[1] % l // 2]

        i_max = image.shape[0] // l
        j_max = image.shape[1] // l

        for i in range(i_max):
            for j in range(j_max):
                image_tile = fastai.vision.Image(
                    torch.tensor(image[l * i:l * (i + 1), l * j:l * (j + 1)],
                                 dtype=torch.float32).unsqueeze(0)
                )
                segment_tile = learn.predict(image_tile)[0]._px.squeeze().numpy()

                if i==0 and j==0:
                    segment = segment_tile
                elif i==0 and j!=0:
                    segment = np.concatenate((segment, segment_tile), axis=1)
                elif i!=0 and j==0:
                    row = segment_tile
                elif i!=0 and j!=0:
                    row = np.concatenate((row, segment_tile), axis=1)

            #After an entire row finishes
            if i!=0:
                segment = np.concatenate((segment, row), axis=0)
        segment_path = output_dir/(image_path.stem + "_segment" + image_path.suffix)
        segment[segment == 1] = 255
        cv.imwrite(segment_path.as_posix(), segment)
        import pdb; pdb.set_trace()

    print(args)