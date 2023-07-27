import unittest
import numpy as np
from multifocal_stitching.stitching import *
from multifocal_stitching.utils import *

def coord_is_close(res, val, tol=5):
    assert np.linalg.norm(np.array(res.coord) - np.array(val), 1) <= tol

class TestStitching(unittest.TestCase):
    def setUp(self):
        parser = add_stitching_args(add_merge_args(get_default_parser()))
        self.args = parser.parse_args(['tests/imgs'])

    def stitch_name(self, name):
        names = [f'{name}_{ext}_small.jpg' for ext in '12']
        img_names = [get_full_path(self.args, name) for name in names]
        res = stitch(self.args, *map(read_img, img_names))
        res_dir = get_full_path(self.args, self.args.result_dir, mkdir=True)
        dx, dy = res.coord
        merge_imgs(self.args, res_dir, names[0], names[1], dx, dy)
        return res

    def test_stitching_high_freq_features(self):
        res = self.stitch_name('high_freq_features')
        coord_is_close(res, (2474,495))
        self.assertAlmostEqual(res.corr_coeff, 0.5548805792236229)

    def test_stitching_large_overlap(self):
        res = self.stitch_name('large_overlap')
        coord_is_close(res, (0,0))
        self.assertAlmostEqual(res.corr_coeff, 0.64077889966183)

    def test_stitching_low_freq_features(self):
        res = self.stitch_name('low_freq_features')
        coord_is_close(res, (2169,667))
        self.assertAlmostEqual(res.corr_coeff, 0.7069386184075431)

    def test_stitching_sharp_blur_overlap(self):
        res = self.stitch_name('sharp_blur_overlap')
        coord_is_close(res, (2897, 96))
        self.assertAlmostEqual(res.corr_coeff, 0.29432216768989594)

    def test_stitching_small_overlap(self):
        res = self.stitch_name('small_overlap')
        coord_is_close(res, (2786, 795))
        self.assertAlmostEqual(res.corr_coeff, 0.709317167243965)

if __name__ == '__main__':
    unittest.main()
