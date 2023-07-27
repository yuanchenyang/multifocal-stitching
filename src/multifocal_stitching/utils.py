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

def get_full_path(args, filename, mkdir=False):
    path = os.path.join(args.dir, filename)
    if mkdir and not os.path.exists(path):
        os.makedirs(path)
    return path

def get_name(path):
    return os.path.split(path)[-1]

def round_int(s):
    return int(round(float(s),0))

def round_int_np(x):
    return np.round(x).astype('int')

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        help='Base directory')
    parser.add_argument("-v", "--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    return parser

def read_img(filename):
    assert os.path.isfile(filename), f'Fild not found: {filename}'
    return np.float64(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
