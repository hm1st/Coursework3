from category_path import get_category, get_path
from dsift import get_dsift
import torch
# from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
# Parameters
NUM_CLUSTERS = 500

category = get_category()
flag =0
for c in category:
    train_path = get_path('train', c)
    for path in train_path:
        image = Image.open(path)
        dsift = get_dsift(image)
        dsift = np.nan_to_num(dsift)
        if flag ==0:
            data = np.zeros(shape=(dsift.shape[0],dsift.shape[1]))
            data[:,:] = dsift
            flag = 1
        else:
            data = np.append(data,dsift,axis=0)
        del dsift
    print('{} Completed!'.format(c))

print(data.shape)
data = torch.tensor(data[np.random.choice(range(data.shape[0]), int(np.floor(0.1*data.shape[0])), replace=False), :])
cluster_ids_x, cluster_centers= kmeans(X=data,num_clusters=NUM_CLUSTERS,device=torch.device('cuda:0'))

# kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(data.reshape((nx*ny,nz)))


np.save('vocab_dsift.npy', cluster_centers)