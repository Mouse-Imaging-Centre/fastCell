import argparse
import tempfile
import fastai.vision
import torch
import cv2 as cv
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Segment the cells from an image.')
parser.add_argument("--image", dest="image", type=str, required=True,
                    help = "Image to segment")
parser.add_argument("--learner", dest="learner", type=str, required=True,
                   help = "Load the Learner object that was saved from export().")
parser.add_argument("--segment-output", dest="segment_output", type=str, required=True)
parser.add_argument("--image-output", dest="image_output", type=str, required=False,
                    help = "Write out the image. It may have been cropped or otherwise processed.")
parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=False,
                   help = "Load the Learner object on the gpu instead of the cpu.")
parser.add_argument("--crop-edges", dest="crop_edges", action="store_true", default=True,
                   help = "Crop the edges if the image is not divisible into 224*224 tiles.")
parser.add_argument("--temp-dir", dest="temp_dir", type=str, default=None)
parser.add_argument("--keep-temp", dest="keep_temp", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

args = parser.parse_args()

l = 224

if __name__ == '__main__':
    fastai.torch_core.defaults.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    #this has to be done because of how stupidly learner() takes its arguments
    learner_path = Path(args.learner)
    learn = fastai.basic_train.load_learner(path = learner_path.parent, file = learner_path.name)

    if args.keep_temp:
        temp_dir = Path(tempfile.mkdtemp(dir=args.temp_dir))
    else:
        temp_object = tempfile.TemporaryDirectory(dir=args.temp_dir)
        temp_dir = Path(temp_object.name)

    image_path=Path(args.image)
    image = cv.imread(args.image, cv.COLOR_BGR2GRAY)

    if args.crop_edges:
        if args.verbose:
            print("Dimensions: " + str(image.shape[0]) + ", " + str(image.shape[1]))
        image = image[image.shape[0] % l // 2:image.shape[0] - image.shape[0] % l // 2,
                image.shape[1] % l // 2:image.shape[1] - image.shape[1] % l // 2]

    i_max = image.shape[0] // l
    j_max = image.shape[1] // l
    if args.verbose:
        print("Segmentation window is " + str(l) + "*" + str(l))
        print("Image has " + str(i_max) + " rows and " + str(j_max) + " columns of image tiles.")

    for i in range(i_max):
        for j in range(j_max):
            if args.verbose:
                print("Working on tile in row:" + str(i+1) + ", column:" + str(j+1) + ".")

            # this is annoying, but a low-risk way to get fastai to read the image properly
            image_tile = image[l * i:l * (i + 1), l * j:l * (j + 1)]
            image_tile_path = temp_dir / (image_path.stem + "_i" + str(i) + "_j" + str(j) + image_path.suffix)
            cv.imwrite(image_tile_path.as_posix(), image_tile)
            image_tile = fastai.vision.open_image(image_tile_path.as_posix())

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
    segment[segment == 1] = 255
    cv.imwrite(args.segment_output, segment)

    if args.image_output:
        cv.imwrite(args.image_output, image)
