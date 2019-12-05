import argparse

def add_postprocessing_group(master_parser: argparse.ArgumentParser) -> None:
    postprocessing_group = master_parser.add_argument_group("Postprocessing")
    postprocessing_group.add_argument("--cell-min-area", dest="cell_min_area", type=int, default=15,
                                       help = """
                                       The neural network provided to --learner may mistakenly segment
                                       stray pixels as cells. All segmented cells with area less than the
                                       value specified by --cell-min-area will be ignored.
                                       """)
    postprocessing_group.add_argument("--process-clusters", dest="process_clusters", action="store_true", default=False,
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
    postprocessing_group.add_argument("--cell-max-area", dest="cell_max_area", type=int,
                        help = "This is only used by --process-clusters")
    postprocessing_group.add_argument("--cell-mean-area", dest="cell_mean_area", type=float,
                        help = "This is only used by --process-clusters")

def validate_postprocessing_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.process_clusters == True:
        if args.cell_max_area is None or args.cell_mean_area is None:
            raise parser.error("--cell-max-area and --cell-mean-area are required for --process-clusters")