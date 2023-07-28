import argparse
import os
import glob
from itertools import tee

import cv2
import numpy as np

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_filenames(args):
    return glob.glob(os.path.join(args.dir, f'*{args.ext}'))

def get_full_path(base_dir, filename, mkdir=False):
    path = os.path.join(base_dir, filename)
    if mkdir and not os.path.exists(path):
        os.makedirs(path)
    return path

def get_name(path):
    return os.path.split(path)[-1]

def round_int_np(x):
    return np.round(x).astype('int')

def read_img(filename):
    assert os.path.isfile(filename), f'Fild not found: {filename}'
    return np.float64(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        help='Base directory')
    parser.add_argument("-v", "--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    parser.add_argument('--ext',
                        help='Filename extension of images',
                        default='.jpg')
    parser.add_argument('--imgs', nargs="+", type=str,
                        help='Stitch only provided images in provided order, otherwise '
                        'will run in batch mode over all images in directory',
                        default=None)
    parser.add_argument('--no_merge',
                        help='Disable generating merged images',
                        action='store_true')
    parser.add_argument('--workers', type=int,
                        help='Number of CPU threads to use in FFT',
                        default=2)
    parser.add_argument('--min_overlap', type=float,
                        help='Set lower limit for overlapping region as a fraction of total image area',
                        default=0.125)
    parser.add_argument('--early_term_thresh', type=float,
                        help='Stop searching when correlation is above this value',
                        default=0.7)
    parser.add_argument('--use_wins', nargs="+", type=int,
                        help='Whether to try using Hanning window',
                        default=(0,))
    parser.add_argument('--peak_cutoff_std', type=float,
                        help='Number of standard deviations below max value to use for peak finding',
                        default=1)
    parser.add_argument('--peaks_dist_threshold', type=float,
                        help='Distance to consider as part of same cluster when finding peak centroid',
                        default=25)
    parser.add_argument('--filter_radii', nargs="+", type=int,
                        help='Low-pass filter radii to try, smaller matches coarser/out-of-focus features',
                        default=(100,50,20))
    parser.add_argument('--stitching_result',
                        help='Stitching result csv file',
                        default='stitching_result.csv')
    parser.add_argument('--result_dir',  type=str,
                        help='Directory to save merged files',
                        default='merged')
    parser.add_argument('--resize_factor', type=int,
                        help='Whether to resize the images saved by a factor',
                        default=1)
    parser.add_argument('--save_gif',
                        help='Whether to save a gif alternating between the merged files',
                        action="store_true")
    return parser
