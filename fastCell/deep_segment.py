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
from fastCell.core import add_default_groups, parse_and_validate
from fastCell.preprocessing import add_preprocessing_group
from fastCell.segmentation import add_segmentation_group
from fastCell.postprocessing import add_postprocessing_group, validate_postprocessing_args

parser = argparse.ArgumentParser(description = 'Segment the cells from an image.')
for add_group in [add_default_groups, add_preprocessing_group, add_segmentation_group, add_postprocessing_group]:
    add_group(parser)

args = parse_and_validate(parser, [validate_postprocessing_args])

l = 224

if __name__ == '__main__':
    fastai.torch_core.defaults.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    #this has to be done because of how stupidly learner() takes its arguments
    learner_path = Path(args.learner)
    learn = fastai.basic_train.load_learner(path = learner_path.parent, file = learner_path.name)
    if args.fp32: learn.to_fp32()

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
            image_tile = fastai.vision.open_image(image_tile_path)

            segment_tile = learn.predict(image_tile)[0].data.squeeze().numpy()

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
    # cv.findContours returns a list of np.ndarray of shape [px, unknown, 2].
    contours, hierarchy = cv.findContours(segment.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
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
                      color=args.centroid_intensity,
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