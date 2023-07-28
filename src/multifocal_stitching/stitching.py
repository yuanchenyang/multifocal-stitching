import math
import csv
import cv2
from collections import namedtuple
from itertools import product
from scipy import fft
from sklearn.cluster import AgglomerativeClustering
from typing import Any, Tuple, Generator

from .utils import *

def get_filter_mask(img: np.ndarray, r: int) -> np.ndarray:
    x, y = img.shape
    mask = np.zeros((x, y), dtype="uint8")
    cv2.circle(mask, (y//2, x//2), r, 255, -1)
    return mask

def apply_filter(fft: Any, img: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    res = fft.fftshift(img)
    res[filter_mask == 0] = 0
    return fft.ifftshift(res)

def corr(a1: np.ndarray, a2: np.ndarray) -> float:
    if len(a1) == 0 or len(a2) == 0:
        return 0
    return np.corrcoef(a1, a2)[0,1]

def get_overlap(img1: np.ndarray, img2: np.ndarray,
                coords: Tuple[int], min_overlap: float=0.) -> Tuple[float,int]:
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

def centroids(coords: np.ndarray, labels: np.ndarray) -> Generator[np.ndarray, None, None]:
    for c in range(labels.max()+1):
        yield round_int_np(coords[labels == c].mean(axis=0))

def get_peak_centroids(res: np.ndarray,
                       peak_cutoff_std: float,
                       peaks_dist_threshold: float) -> Generator[np.ndarray, None, None]:
    ''' Cluster peaks that are within `peak_cutoff_std` standard deviations
    below maximum peak, then yields centroid of clusters
    '''
    cutoff = res > (res.max() - peak_cutoff_std * res.std())
    if cutoff.sum() > 2:
        X = np.argwhere(cutoff)
        labels = AgglomerativeClustering(
            n_clusters=None,
            linkage='single',
            distance_threshold=peaks_dist_threshold
        ).fit(X).labels_
        cents = list(centroids(X, labels))
        yield from sorted(cents, key=lambda coord: res[tuple(coord)])
    else:
        yield from np.argwhere(cutoff)

StitchingResult = namedtuple(
    'StitchingResult',
    ['corr_coeff', 'corr', 'coord', 'val', 'area', 'best_r', 'best_win']
)

def print_stitching_result(r: StitchingResult):
    dx, dy = r.coord
    print(f'dx:{dx: 5} dy:{dy: 5} corr:{r.corr_coeff:+f} area:{r.area: 9} r:{r.best_r: 3}')

def candidate_stitches(img1: np.ndarray, img2: np.ndarray,
                       use_wins: Tuple[int] = (0,),
                       workers: int = 2,
                       peak_cutoff_std: float = 1,
                       peaks_dist_threshold: float = 25,
                       filter_radii: Tuple[int] = (100,50,20),
                       min_overlap: float = 0.125,
                       early_term_thresh: float = 0.7,
                       verbose: bool = False,
                       ) -> Generator[StitchingResult, None, None]:
    assert img1.shape == img2.shape, 'Images must be of same size!'
    assert len(img1.shape) == 2, 'Image must be 2D array (one color channel)'
    Y, X = img1.shape

    # Create window if required
    if 1 in use_wins:
        win = cv2.createHanningWindow(img1.T.shape, cv2.CV_64F)

    for use_win in use_wins:
        # 1. Compute FFT of input images
        f1, f2 = [fft.fft2(img * win if use_win else img,
                           norm='ortho', workers=workers)
                  for img in (img1, img2)]

        for r in filter_radii:
            # 2. Apply low-pass filter to images in frequency domain
            mask = get_filter_mask(img1, r)
            G1, G2 = [apply_filter(fft, f, mask) for f in (f1, f2)]

            # 3. Compute cross power spectrum
            R = G1 * np.ma.conjugate(G2)
            R /= np.absolute(R)

            # 4. Obtain cross correlation in spatial domain by taking inverse FFT
            res = fft.ifft2(R, img1.shape, norm='ortho', workers=workers)

            # 5. Group peaks and find centroids of groups
            for dy, dx in get_peak_centroids(res, peak_cutoff_std, peaks_dist_threshold):
                for dX, dY in product((dx, -X+dx), (dy, -Y+dy)):
                    coef, area = get_overlap(img1, img2, (dX, dY), min_overlap=min_overlap)
                    result = StitchingResult(coef, res, (dX, dY), res[dY, dX], area, r, use_win)
                    if verbose:
                        print_stitching_result(result)
                    yield result
                    if coef >= early_term_thresh:
                        return

def stitch(img1: np.ndarray, img2: np.ndarray, **kwargs) -> StitchingResult:
    return max(candidate_stitches(img1, img2, **kwargs), key=lambda r: r.corr_coeff)
