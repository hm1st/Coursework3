import numpy as np

def get_tiny(image):
    image_resize = np.array(image.resize((16,16), box=(128-20,128-20,128+20,128+20))).flatten()   #Simple crop around the center and resize to 16*16  .crop((128-20,128-20,128+20,128+20))
    image_norm = (image_resize-np.mean(image_resize))/np.std(image_resize)
    return np.array(image_norm)
    