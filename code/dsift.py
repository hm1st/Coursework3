import cyvlfeat
import numpy as np

def get_dsift(image):
    if image.size != (256,256):
        image = image.resize((256,256))
    _, descriptors = cyvlfeat.sift.dsift(np.array(image), step=[4,4], fast=True)
    return descriptors