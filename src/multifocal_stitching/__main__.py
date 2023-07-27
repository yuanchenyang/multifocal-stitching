import csv
from .stitching import add_stitching_args, stitch
from .utils import get_default_parser, get_filenames, get_name, get_full_path, pairwise, read_img
from .merge_imgs import add_merge_args, merge_imgs

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
