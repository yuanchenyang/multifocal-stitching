import gradio as gr
import numpy as np
import cv2
from multifocal_stitching import candidate_stitches, stitch, merge

def stitch_interface(img1, img2):
                     #use_wins, workers, peak_cutoff_std,
                     #peaks_dist_threshold, filter_radii, min_overlap,
                     #early_term_thresh):
    res = stitch(*[cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY) for im in (img1, img2)])
    dx, dy = res.coord
    return merge(img1, img2, dx, dy, resize_factor=8)

demo = gr.Interface(fn=stitch_interface, inputs=[
    gr.Image(type='pil'), gr.Image(type='pil')
], outputs=[
    gr.Gallery()
], examples=[
    ["tests/imgs/high_freq_features_1_small.jpg",
     "tests/imgs/high_freq_features_2_small.jpg", ]
])






demo.launch()
