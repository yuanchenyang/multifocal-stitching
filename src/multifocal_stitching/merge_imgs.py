import os
from PIL import Image
from .utils import *

def add_merge_args(parser):
    parser.add_argument('-s', '--stitching_result',
                        help='Stitching result csv file',
                        default='stitching_result.csv')
    parser.add_argument('-d', '--result_dir',
                        help='Directory to save merged files',
                        default='merged')
    parser.add_argument('-r', '--exclude_reverse',
                        help='Whether to additionally include img2 on top of img1',
                        action='store_true')
    return parser

def merge_imgs(args, res_dir, img1, img2, dx, dy):
    if args.verbose:
        print('Merging:', img1, img2)
    i1, i2 = [Image.open(get_full_path(args,img)) for img in (img1, img2)]
    dx, dy = map(round_int, (dx, dy))
    W, H = i1.size
    new_W, new_H = W + abs(dx), H + abs(dy)
    i1_x = -dx if dx < 0 else 0
    i1_y = -dy if dy < 0 else 0
    i2_x = dx if dx > 0 else 0
    i2_y = dy if dy > 0 else 0
    res = Image.new(mode='RGB', size=(new_W, new_H))
    res.paste(i1, (i1_x, i1_y))
    res.paste(i2, (i2_x, i2_y))
    res_path = os.path.join(res_dir,
        f'{os.path.splitext(img1)[0]}__{os.path.splitext(img2)[0]}.jpg')
    res.save(res_path)
    if not args.exclude_reverse:
        res.paste(i1, (i1_x, i1_y))
        res.save(res_path[:-4] + '_r.jpg')

def main():
    parser = add_merge_args(get_default_parser())
    args = parser.parse_args()
    res_dir = get_full_path(args, args.result_dir, mkdir=True)
    with open(get_full_path(args, args.stitching_result)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header row
        for img1, img2, dx, dy, *_ in reader:
            merge_imgs(args, res_dir, img1, img2, dx, dy)

if __name__=='__main__':
    main()
