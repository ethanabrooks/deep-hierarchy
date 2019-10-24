#! /usr/bin/env python
import imageio
import numpy as np
import skimage.draw
import skimage.transform

np.set_printoptions(linewidth=1000)
size = 100
img = np.zeros((size + 1, size + 1))
scale = size // 10
mid = size // 2
for start in [0, 3, 5, 8]:
    a = start * scale
    b = (start + 2) * scale
    rr, cc, val = skimage.draw.line_aa(a, mid, b, mid)
    img[rr, cc] = val
    rr, cc, val = skimage.draw.line_aa(mid, a, mid, b)
    img[rr, cc] = val
rr, cc, val = skimage.draw.line_aa(50, 25, 25, 50)
img[rr, cc] = val
# img = skimage.transform.rescale(img, 0.25)
imageio.imwrite("/tmp/img.png", img)
