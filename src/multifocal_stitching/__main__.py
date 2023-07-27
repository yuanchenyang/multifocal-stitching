import csv
from .stitching import add_stitching_args, stitch
from .utils import get_default_parser, get_filenames, get_name, get_full_path, pairwise, read_img
from .merge_imgs import add_merge_args, merge_imgs

CSV_HEADER = ['Img 1', 'Img 2', 'X offset', 'Y offset', 'Corr Value', 'Area', 'r', 'use_win']

def main():
    parser = add_stitching_args(add_merge_args(get_default_parser()))
    args = parser.parse_args()
    img_names = sorted(get_filenames(args))
    with open(get_full_path(args, args.stitching_result), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(CSV_HEADER)
        for img_names in pairwise(img_names):
            if args.verbose:
                print('Stitching', *img_names)
            result = stitch(*map(read_img, img_names),
                            use_wins = args.use_wins,
                            workers = args.workers,
                            peak_cutoff_std = args.peak_cutoff_std,
                            peaks_dist_threshold = args.peaks_dist_threshold,
                            filter_radii = args.filter_radii,
                            min_overlap = args.min_overlap,
                            early_term_thresh = args.early_term_thresh,
                            verbose = args.verbose)
            dx, dy = result.coord
            img_name1, img_name2 = map(get_name, img_names)
            writer.writerow([img_name1, img_name2, dx, dy, result.corr_coeff,
                             result.area, result.best_r, result.best_win])
            if not args.no_merge:
                res_dir = get_full_path(args, args.result_dir, mkdir=True)
                merge_imgs(args, res_dir, img_name1, img_name2, dx, dy)

if __name__=='__main__':
    main()
