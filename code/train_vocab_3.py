from category_path import get_category, get_path
from dsift import get_dsift
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

# Parameters
NUM_CLUSTERS = 500

category = get_category()
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
for c in category:
    train_path = get_path('train', c)
    for path in train_path:
        image = Image.open(path)
        dsift = get_dsift(image)
        dsift = np.nan_to_num(dsift)
        kmeans.fit(dsift)
        del dsift
    print('{} Completed!'.format(c))
vocab = kmeans.cluster_centers_
np.save('vocab_dsift.npy', vocab)