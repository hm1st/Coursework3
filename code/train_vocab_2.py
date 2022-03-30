from category_path import get_category, get_path
from pixels_feature import get_feature
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

NUM_CLUSTERS = 500
SAMPLE_SIZE = 500

category = get_category()
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
idx = np.random.randint(low=0, high=3969, size=SAMPLE_SIZE)
for c in category:
    train_path = get_path('train', c)
    for path in train_path:
        image = Image.open(path)
        features = get_feature(image)
        samples = features[idx]
        samples = np.nan_to_num(samples)
        kmeans.fit(samples)
        del features
        del samples
    print('{} Completed!'.format(c))
vocab = kmeans.cluster_centers_
np.save('vocab_pixels.npy', vocab)