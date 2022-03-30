from unicodedata import category
from get_tiny_images import get_tiny
from category_path import get_category, get_path
import numpy as np
from PIL import Image
import os
from glob import glob
from sklearn.neighbors import KNeighborsClassifier

train_images = []
train_labels = []
category = get_category()
for c in category:
    train_path = get_path('train', c)
    for path in train_path:
        img = Image.open(path)
        tiny = get_tiny(img)
        train_images.append(tiny)
        train_labels.append(c)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

model_knn = KNeighborsClassifier(n_neighbors=15, leaf_size=100)
model_knn.fit(train_images, train_labels)
print('KNN model training completed! Begin testing...')

testing_path = r'./testing'
with open('./run1.txt', 'w') as file:
    for i in range(2988):
        test_path = glob(os.path.join(testing_path, '{}.jpg'.format(i)))
        if test_path != []:
            image = Image.open(test_path[0])
            image_resize = np.array(image.resize((16,16), box=(128-20,128-20,128+20,128+20))).flatten()    #Simple crop around the center and resize to 16*16
            image_norm = (image_resize-np.mean(image_resize))/np.std(image_resize)
            res = model_knn.predict(image_norm.reshape(1,-1))
            file.write(str('{}.jpg'.format(i) + ' ' + res[0] + '\n'))
        if i%500 == 0:
            print('{} prediction completed!'.format(i))
    file.close()