import unittest
import numpy as np
import os
import sys
import shutil
import csv
from multifocal_stitching.stitching import stitch
from multifocal_stitching.utils import read_img, get_full_path
from multifocal_stitching.merge_imgs import merge_and_save
from multifocal_stitching.__main__ import main as cli
from multifocal_stitching.__main__ import CSV_HEADER

def coord_is_close(res, val, tol=5):
    assert np.linalg.norm(np.array(res.coord) - np.array(val), 1) <= tol

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.base_dir = 'tests/img_folder'
        shutil.rmtree(self.base_dir, ignore_errors=True)
        os.makedirs(self.base_dir)
        shutil.copy('tests/imgs/high_freq_features_1_small.jpg', self.base_dir)
        shutil.copy('tests/imgs/high_freq_features_2_small.jpg', self.base_dir)
        sys.argv = ['', self.base_dir]

    def test_cli(self):
        cli()
        csvfilename = get_full_path(self.base_dir, 'stitching_result.csv')
        self.assertTrue(os.path.isfile(csvfilename))
        with open(csvfilename) as csvfile:
            reader = csv.reader(csvfile)
            self.assertEqual(next(reader), CSV_HEADER)
            img_name1, img_name2, dx, dy, corr_coeff, area, best_r, best_win = next(reader)
            self.assertEqual(img_name1, 'high_freq_features_1_small.jpg')
            self.assertEqual(img_name2, 'high_freq_features_2_small.jpg')
            self.assertAlmostEqual(int(dx), 2474, delta=2)
            self.assertAlmostEqual(int(dy), 495, delta=2)
            self.assertAlmostEqual(float(corr_coeff), 0.5548805792236229)
            self.assertEqual(int(area), 2274390)
            self.assertEqual(int(best_r), 50)
            self.assertEqual(int(best_win), 0)

        merged_name, merged_r_name = [os.path.join(
            self.base_dir, 'merged',
            f'high_freq_features_1_small__high_freq_features_2_small_{i}.jpg')
                        for i in range(2)]
        self.assertTrue(os.path.isfile(merged_name))
        self.assertTrue(os.path.isfile(merged_r_name))
        merged = read_img(merged_name)
        merged_r = read_img(merged_r_name)
        self.assertEqual(merged.shape, merged_r.shape)
        self.assertEqual(merged.shape, (2655, 6314))

class TestStitch(unittest.TestCase):
    def setUp(self):
        self.base_dir = 'tests/imgs'

    def stitch_name(self, name):
        names = [f'{name}_{ext}_small.jpg' for ext in '12']
        res = stitch(*[read_img(get_full_path(self.base_dir, name)) for name in names])
        res_dir = get_full_path(self.base_dir, 'merged', mkdir=True)
        dx, dy = res.coord
        merge_and_save(self.base_dir, res_dir, names[0], names[1], dx, dy,
                       resize_factor=8, save_gif=True)
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
