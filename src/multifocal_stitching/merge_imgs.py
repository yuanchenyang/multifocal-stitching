import os
from PIL import Image
from typing import Tuple
from .utils import *

def merge(i1: Image, i2: Image, dx:int, dy:int) -> Tuple[Image, Image]:
    assert i1.size == i2.size, "Images must be same size!"
    W, H = i1.size
    new_W, new_H = W + abs(dx), H + abs(dy)

    i1_x = -dx if dx < 0 else 0
    i1_y = -dy if dy < 0 else 0
    i2_x = dx if dx > 0 else 0
    i2_y = dy if dy > 0 else 0

    res = Image.new(mode='RGB', size=(new_W, new_H))
    res.paste(i1, (i1_x, i1_y))
    res.paste(i2, (i2_x, i2_y))

    res_r = res.copy()
    res_r.paste(i1, (i1_x, i1_y))

    return res, res_r

def merge_and_save(base_dir:str, res_dir:str, img_name1:str, img_name2:str,
                   dx:int, dy:int, resize_factor:int=1, save_gif:bool=False):
    i1, i2 = [Image.open(get_full_path(base_dir, i)) for i in (img_name1, img_name2)]
    res = merge(i1, i2, dx, dy)
    W, H = res[0].size
    res_resized = [r.resize((W // resize_factor, H // resize_factor), Image.LANCZOS)
                   for r in res]
    for i, r in enumerate(res_resized):
        base1, base2 = [os.path.splitext(n)[0] for n in (img_name1, img_name2)]
        r.save(get_full_path(res_dir, f'{base1}__{base2}_{i}.jpg'))
    if save_gif:
        res_resized[0].save(get_full_path(res_dir, f'{base1}__{base2}.png'), save_all=True,
                            append_images=res_resized[1:], duration=500, loop=0)
