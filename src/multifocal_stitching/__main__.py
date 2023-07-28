import csv
from .stitching import stitch
from .utils import get_default_parser, get_filenames, get_name, get_full_path, pairwise, read_img
from .merge_imgs import merge_and_save

CSV_HEADER = ['Img 1', 'Img 2', 'X offset', 'Y offset', 'Corr Value', 'Area', 'r', 'use_win']

def main():
    args = get_default_parser().parse_args()
    if args.imgs is not None:
        assert len(args.imgs) >= 2, 'Can only stitch two or more images!'
        img_names = [get_full_path(args.dir, img) for img in args.imgs]
    else:
        img_names = sorted(get_filenames(args))
    with open(get_full_path(args.dir, args.stitching_result), 'w') as outfile:
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
                if args.verbose:
                    print('Merging:', img_name1, img_name2)
                res_dir = get_full_path(args.dir, args.result_dir, mkdir=True)
                merge_and_save(args.dir, res_dir, img_name1, img_name2, dx, dy,
                               resize_factor=args.resize_factor,
                               save_gif=args.save_gif,)

if __name__=='__main__':
    main()
