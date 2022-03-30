import numpy as np

def get_feature(image):
    step=4
    if image.size != (256,256):
        image = image.resize((256,256))
    feature = np.zeros((63,63,64))
    image = np.array(image)
    for i in range(63):
        for j in range(63):
            vector = image[i*step:i*step+8, j*step:j*step+8].flatten()
            vector = (vector - np.mean(vector))/np.std(vector)
            feature[i, j] = vector
    feature = np.nan_to_num(feature)
    return feature.reshape((-1,64))