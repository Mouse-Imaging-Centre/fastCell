import argparse, re
import tempfile
import fastai.vision
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from fastCell.core import random_cells

from pathlib import Path
from fastCell.Hcolumns import *

parser = argparse.ArgumentParser(description='Segment the cells from an image.')
parser.add_argument("--image", dest="image", type=str, required=True,
                    help = "Image to segment")
parser.add_argument("--learner", dest="learner", type=str, required=True,
                   help = "Load the Learner object that was saved from export().")

parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=False,
                   help = "Load the Learner object on the gpu instead of the cpu.")
parser.add_argument("--crop-edges", dest="crop_edges", action="store_true", default=True,
                   help = "Crop the edges if the image is not divisible into 224*224 tiles.")
parser.add_argument("--cell-min-area", dest="cell_min_area", type=int, default=15,
                   help = """
                   The neural network provided to --learner may mistakenly segment
                   stray pixels as cells. All segmented cells with area less than the
                   value specified by --cell-min-area will be ignored.
                   """)
parser.add_argument("--temp-dir", dest="temp_dir", type=str, default=None,
                    help="Please do not use any parent directory notation like .. on UNIX systems.")
parser.add_argument("--keep-temp", dest="keep_temp", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
parser.add_argument("--segment-intensity", dest="segment_intensity", type=int, default=255)

cluster_group = parser.add_argument_group("Cluster Processing")
cluster_group.add_argument("--process-clusters", dest="process_clusters", action="store_true", default=False,
                    help = """
                    Individual cells that aren't touching are processed by being reduced to
                    a single point at their centroid. Clusters of cells area identified by the
                    maximum area criterium specified by --cell-max-area. The number of cells 
                    contained in the cluster is found by dividing the cluster's total area by 
                    the mean area criterium specified by --cell-mean-area. That many cell centroids
                    are randomly and uniformly sampled inside the cluster. Note that this will not
                    work out of the box. Your neural network provided to --learner must be trained
                    to recognize clusters of cells in addition to individual cells.
                    """)
cluster_group.add_argument("--cell-max-area", dest="cell_max_area", type=int, default=None,
                    help = "This is only used by --process-clusters")
cluster_group.add_argument("--cell-mean-area", dest="cell_mean_area", type=float, default=None,
                    help = "This is only used by --process-clusters")

output_group = parser.add_argument_group("Outputs")
output_group.add_argument("--segment-output", dest="segment_output", type=str, required=False,
                    help="Write out the segmentation.")
output_group.add_argument("--centroids-output", dest="centroids_output", type=str, required=False,
                    help="Write out each cell as pixel.")
output_group.add_argument("--outlines-output", dest="outlines_output", type=str, required=False,
                    help="Outline the identified cells.")
output_group.add_argument("--image-output", dest="image_output", type=str, required=False,
                    help = "Write out the image. It may have been cropped or otherwise processed.")
args = parser.parse_args()

if (args.process_clusters and not (args.cell_max_area and args.cell_mean_area)):
    parser.error("to --process-clusters, I need to know the --cell-max-area and --cell-mean-area")

l = 224

if __name__ == '__main__':
    fastai.torch_core.defaults.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    #this has to be done because of how stupidly learner() takes its arguments
    learner_path = Path(args.learner)
    learn = fastai.basic_train.load_learner(path = learner_path.parent, file = learner_path.name)

    os.makedirs(args.temp_dir, exist_ok=True)
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

    #Segment each tile
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

    #Post-process
    if re.match("^4.1", cv.__version__):
        # cv.findContours returns a list of np.ndarray of shape [px, unknown, 2].
        contours, hierarchy = cv.findContours(segment.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    elif re.match("^3.4", cv.__version__):
        image, contours, hierarchy = cv.findContours(segment.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = [np.squeeze(contour,axis=1) for contour in contours]

    df = pd.DataFrame({'contour': contours}).assign(
        moments = lambda df: df.contour.apply(lambda contour: cv.moments(contour)),
        area = lambda df: df.contour.apply(lambda contour: cv.contourArea(contour)),
        perimeter = lambda df: df.contour.apply(lambda contour: cv.arcLength(contour, closed=True))
    )

    df = df[df.area > args.cell_min_area]

    if df.empty:
        df["centroid"] = None
    else:
        if args.process_clusters:
            df["centroid"] = df.apply(
                lambda row: random_cells(row.contour, args.cell_mean_area) if row.area > args.cell_max_area else
                [(int(row.moments['m10'] / row.moments['m00']), int(row.moments['m01'] / row.moments['m00']))],
                axis = 1
            )
            df = df.explode("centroid")
        else:
            df = df.assign(
                centroid = lambda df: df.moments.apply(lambda moments:
                                                      (int(moments['m10'] / moments['m00']),
                                                       int(moments['m01'] / moments['m00']))
                                                       )
            )

    #Write outputs
    if args.segment_output:
        segment = np.zeros(image.shape, np.uint8)
        cv.drawContours(image=segment,
                        contours=df.contour.tolist(),
                        contourIdx=-1, #negative value means draw all contours
                        color=args.segment_intensity,
                        thickness=-1) #negative value means fill it in
        cv.imwrite(args.segment_output, segment)

    if args.centroids_output:
        centroids = np.zeros(image.shape, np.uint8)
        for centroid in df.centroid:
            cv.circle(img=centroids,
                      center=centroid,
                      radius=0,
                      color=args.segment_intensity,
                      thickness=1)
        cv.imwrite(args.centroids_output, centroids)

    if args.outlines_output:
        outlines = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        cv.drawContours(image=outlines,
                        contours=df.contour.tolist(),
                        contourIdx=-1,  # negative value means draw all contours
                        color=(0,255,0),
                        thickness=1,
                        lineType = cv.LINE_AA)
        cv.imwrite(args.outlines_output, outlines)

    if args.image_output:
        cv.imwrite(args.image_output, image)
