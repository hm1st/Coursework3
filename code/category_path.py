import os
from glob import glob

def get_category():
    training_path = r'./training'
    category = os.listdir(training_path)
    if category.__contains__('.DS_Store'):
        category.remove('.DS_Store')
    return category

def get_path(t='train', c='Forest'):
    training_path = r'./training'
    testing_path = r'./testing'
    if t == 'train':
        return glob(os.path.join(training_path, c, '*.jpg'))
    if t == 'test':
        return glob(os.path.join(testing_path, '*.jpg'))

if __name__ == '__main__':
    print(get_path('train', 'store'))