from pixels_feature import get_feature
from dsift import get_dsift
from sklearn.cluster import KMeans

def get_bow(image, t, vocab):
    kmeans_clf = KMeans(n_clusters=vocab.shape[0])
    kmeans_clf.cluster_centers_ = vocab
    if t == 'pixels':
        feature_vector = get_feature(image)
    if t == 'dsift':
        feature_vector = get_dsift(image)
    res = kmeans_clf.predict(feature_vector)
    return res