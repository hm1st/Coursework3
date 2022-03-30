from sklearn.svm import NuSVC
import numpy as np
from PIL import Image
from bow import *
from category_path import *
from glob import glob
import os

category = get_category()
X = []
y = []
vocab = np.load('./vocab_dsift.npy')
print('Vocabulary loaded!')
for c in category:
    train_path = get_path('train', c)
    for path in train_path:
        test_image = Image.open(path)
        bow = get_bow(test_image, 'dsift', vocab)
        X.append(bow)
        y.append(c)
    print('{} Completed!'.format(c))

svm = NuSVC()
svm.fit(X, y)
print('Non-linear SVM model training completed! Begin testing...')

testing_path = r'./testing'
with open('./run3.txt', 'w') as file:
    for i in range(2988):
        test_path = glob(os.path.join(testing_path, '{}.jpg'.format(i)))
        if test_path != []:
            image = Image.open(test_path[0])
            bow = get_bow(image, 'dsift', vocab)
            res = svm.predict(bow.reshape(1,-1))
            file.write(str('{}.jpg'.format(i) + ' ' + res[0] + '\n'))
        if i%500 == 0:
            print('{} prediction completed!'.format(i))
    file.close()