import math
import csv
import cv2
from collections import namedtuple
from itertools import product
from scipy import fft
from sklearn.cluster import AgglomerativeClustering

from .utils import *
from .merge_imgs import add_merge_args, merge_imgs

def get_filter_mask(img, r):
    x, y = img.shape
    mask = np.zeros((x, y), dtype="uint8")
    cv2.circle(mask, (y//2, x//2), r, 255, -1)
    return mask

def apply_filter(fft, img, filter_mask):
    res = fft.fftshift(img)
    res[filter_mask == 0] = 0
    return fft.ifftshift(res)

def corr(a1, a2):
    if len(a1) == 0 or len(a2) == 0:
        return 0
    return np.corrcoef(a1, a2)[0,1]

def get_overlap(img1, img2, coords, min_overlap=0.):
    dx, dy = coords
    assert img1.shape == img2.shape
    Y, X = img1.shape
    if dy >= 0 and dx >= 0:
        s1, s2 = img1[dy:Y, dx:X], img2[0:Y-dy, 0:X-dx]
    elif dy < 0 and dx >= 0:
        s1, s2 = img1[0:Y+dy, dx:X], img2[-dy:Y, 0:X-dx]
    else:
        return get_overlap(img2, img1, (-dx, -dy), min_overlap=min_overlap)
    assert s1.shape == s2.shape
    area = s1.shape[0] * s1.shape[1]
    if area < min_overlap*Y*X:
        return -1, area
    f1, f2 = s1.flatten(), s2.flatten()
    return corr(f1, f2), area

def centroids(coords, labels):
    for c in range(labels.max()+1):
        yield round_int_np(coords[labels == c].mean(axis=0))

def get_peak_centroids(args, res):
    #yield round_int_np(np.unravel_index(np.argmax(res), res.shape))
    #cutoff = res > (res.mean() + args.peak_cutoff_std * res.std())
    cutoff = res > (res.max() - args.peak_cutoff_std * res.std())
    if cutoff.sum() > 2:
        X = np.argwhere(cutoff)
        labels = AgglomerativeClustering(
            n_clusters=None,
            linkage='single',
            distance_threshold=args.peaks_dist_threshold
        ).fit(X).labels_
        cents = list(centroids(X, labels))
        yield from sorted(cents, key=lambda coord: res[tuple(coord)])
    else:
        yield from np.argwhere(cutoff)

StitchingResult = namedtuple(
    'StitchingResult',
    ['corr_coeff', 'corr', 'coord', 'val', 'area', 'best_r', 'best_win']
)

def candidate_stitches(args, img1, img2):
    assert img1.shape == img2.shape
    win = cv2.createHanningWindow(img1.T.shape, cv2.CV_64F)
    Y, X = img1.shape
    for use_win in args.use_wins:
        f1, f2 = [fft.fft2(img * win if use_win else img,
                           norm='ortho', workers=args.workers)
                  for img in (img1, img2)]
        for r in args.filter_radius:
            mask = get_filter_mask(img1, r)
            G1, G2 = [apply_filter(fft, f, mask) for f in (f1, f2)]
            R = G1 * np.ma.conjugate(G2)
            R /= np.absolute(R)
            res = fft.ifft2(R, img1.shape, norm='ortho', workers=args.workers)
            for dy, dx in get_peak_centroids(args, res):
                for dX, dY in product((dx, -X+dx), (dy, -Y+dy)):
                    coef, area = get_overlap(img1, img2, (dX, dY),
                                             min_overlap=args.min_overlap)
                    if args.verbose:
                        print(f'dx:{dX: 5} dy:{dY: 5} corr:{coef:+f} area:{area: 9} r:{r: 3}')
                    yield StitchingResult(coef, res, (dX, dY), res[dY, dX], area, r, use_win)
                    if coef >= args.early_term_thresh:
                        return

def stitch(args, img1, img2):
    return max(candidate_stitches(args, img1, img2), key=lambda r: r.corr_coeff)

def add_stitching_args(parser):
    parser.add_argument('--ext',
                        help='Filename extension of images',
                        default='.jpg')
    parser.add_argument('--no_merge',
                        help='Disable generating merged images',
                        action='store_true')
    parser.add_argument('--workers', type=int,
                        help='Number of CPU threads to use in FFT',
                        default=2)
    parser.add_argument('--min_overlap', type=int,
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
    parser.add_argument('--filter_radius', nargs="+", type=int,
                        default=(100,50,20),
                        help='Low-pass filter radii to try, smaller matches coarser/out-of-focus features')
    return parser

def main():
    parser = add_stitching_args(add_merge_args(get_default_parser()))
    args = parser.parse_args()
    img_names = sorted(get_filenames(args))
    with open(get_full_path(args, args.stitching_result), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Img 1', 'Img 2', 'X offset', 'Y offset', 'Corr Value', 'Area', 'r', 'use_win'])
        for img_names in pairwise(img_names):
            if args.verbose: print('Stitching', *img_names)
            corr, res, (dx, dy), val, area, r, use_win = stitch(args, *map(read_img, img_names))
            img_name1, img_name2 = map(get_name, img_names)
            writer.writerow([img_name1, img_name2, dx, dy, corr, area, r, use_win])
            if not args.no_merge:
                res_dir = get_full_path(args, args.result_dir, mkdir=True)
                merge_imgs(args, res_dir, img_name1, img_name2, dx, dy)

if __name__=='__main__':
    main()
