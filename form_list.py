import os
import random
import numpy as np

def form_list(img):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/'+path.split('/')[-1]+'.txt'), 'w') as f:
        img.sort(key=lambda x: x[4:6])
        for idx, i in enumerate(img):
            f.write(i)
            if idx != len(img)-1 :
                f.write('\n')
# form_list(img)

def random_from_list_train_infer(img):
    initList = list(np.arange(0,len(img),1))

    trainList=random.sample(range(0, len(img)), int(len(img)*0.6))
    trainList.sort()

    sub = 0
    for i in trainList:
        initList.pop(i-sub)
        sub += 1
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/'+path.split('/')[-1]+'-train.txt'), 'w') as f:
        for idx, i in enumerate(trainList):
            f.write(img[i])
            if idx != len(trainList)-1 :
                f.write('\n')

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/'+path.split('/')[-1]+'-infer.txt'), 'w') as f:
        for idx, i in enumerate(initList):
            f.write(img[i])
            if idx != len(initList)-1 :
                f.write('\n')

if __name__ == "__main__":
    path = '/data/spine1.0-nii'
    files = os.listdir(path)
    img = [i for i in files if 'case' in i]
    mask = [i for i in files if 'mask' in i]
    for i,m in zip(img, mask):
        print(i,m)
    
    random_from_list_train_infer(img)