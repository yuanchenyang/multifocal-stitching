import gradio as gr
import numpy as np
import cv2
from matplotlib import pyplot as plt
from multifocal_stitching import candidate_stitches, stitch, merge

def stitch_interface(img1, img2,
                     filter_radii,
                     min_overlap,
                     peak_cutoff_std,
                     peaks_dist_threshold,
                     use_wins):
    grey_imgs = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY) for im in (img1, img2)]
    WINS = [(0,), (1,), (0,1)]
    results = sorted(list(candidate_stitches(*grey_imgs,
                                             early_term_thresh=1.0,
                                             filter_radii=map(int, filter_radii),
                                             min_overlap=min_overlap,
                                             peak_cutoff_std=peak_cutoff_std,
                                             peaks_dist_threshold=peaks_dist_threshold,
                                             use_wins=WINS[use_wins]
                                             )),
                     key=lambda r: r.corr_coeff, reverse=True)
    best = results[0]
    dx, dy = best.delta
    table = [[r.corr_coeff, r.area, r.r, r.use_win, int(r.delta[0]), int(r.delta[1])]
             for r in results]
    fig = plt.figure()
    plt.imshow(abs(best.corr))
    xs, ys = zip(*set(tuple(r.freq_delta) for r in results))
    plt.scatter(xs, ys, marker='x', c='r')
    return merge(img1, img2, dx, dy, resize_factor=8), fig, table

examples = [
    [f'tests/imgs/{name}_1_small.jpg',
     f'tests/imgs/{name}_2_small.jpg',
     ] for name in (
         'high_freq_features',
         'low_freq_features',
         'large_overlap',
         'small_overlap',
         'sharp_blur_overlap',
     )
]
demo = gr.Interface(fn=stitch_interface, inputs=[
    gr.Image(type='pil'), gr.Image(type='pil'),
    gr.Dropdown(list(range(10,501,10)), value=[100, 50, 20], multiselect=True, label='filter_radii',
                info='Low-pass filter radii to try, smaller matches coarser/out-of-focus features'),
    gr.Slider(minimum=0, maximum=1, step=0.001, value=0.125, label='min_overlap',
              info='Set lower limit for overlapping region as a fraction of total image area'),
    gr.Slider(minimum=0, maximum=5, step=0.1, value=1.0, label='peak_cutoff_std',
              info='Number of standard deviations below max value to use for peak finding'),
    gr.Slider(minimum=1, maximum=100, step=1, value=25, label='peaks_dist_threshold',
              info='Distance to consider as part of same cluster when finding peak centroid'),
    gr.Radio(['No window', 'Hanning window', 'Both'], label='use_wins',type='index', value='No window',
             info='Whether to try using Hanning window'),

], outputs=[
    gr.Gallery(label='Merged Images'),
    gr.Plot(label='Frequency domain peaks'),
    gr.DataFrame(label='Candidate Stitches',
                 headers=['Corr Value', 'Area', 'r', 'use_win', 'X offset', 'Y offset'],
                 datatype=['number', 'number', 'number', 'number', 'number', 'bool'],
                 max_rows=10),
], examples=examples)

if __name__ == '__main__':
    demo.launch()
