import os
import torch
import numpy as np

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset,Subset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self,root,classes,target_size:tuple,shuffle=False,**kwargs):
        super().__init__()
        self.root=  root
        self.classes = [str(c) for c in classes]
        self.target_size = target_size if not type(classes) is int else (target_size,target_size)
        self.shuffle = shuffle        

        self.all_dir = kwargs.pop('all_dir',None)
        if self.all_dir is not None:
            if self.all_dir == True:
                temp = np.array(['unknown'])
                self.classes = np.array(os.listdir(self.root))
                self.classes = np.append(temp,self.classes)
                print(self.classes)
        self._load_data()
        self._set_index_array()     
        self._set_shuffle()
    
        # kwargs
        self.samples = kwargs.pop('samples',None)
        if self.samples is not None:
            self._sampling_data()

        

    def _sampling_data(self):
        s = np.random.choice(self.__len__()-self.samples,1).item() if self.shuffle else 0
        e = self.samples
        self.data = self.data[s:e+s]
        self.label = self.label[s:e+s]

    def _load_data(self):
        self.data = []
        self.label = []

        for i,cla in enumerate(self.classes):
            sub_dir = os.path.join(self.root,cla)
            if not os.path.exists(sub_dir):
                print(f'Not found images in "{cla}" directory.')
                continue
            sub_files = os.listdir(sub_dir)
            self.data.extend([os.path.join(sub_dir,sub_file) for sub_file in sub_files])
            l=len(sub_files)
            self.label.extend([i for _ in range(l)])
            
            print(f'Found {l} images in "{cla}" directory.')

        self.data = np.array(self.data) 
        self.label = np.array(self.label)
        

    def _set_index_array(self):
        self.index_array = np.arange(self.__len__())
        if self.shuffle:
            self.index_array = np.random.permutation(self.__len__())

    def _set_shuffle(self):
        self.data = self.data[self.index_array]
        self.label = self.label[self.index_array]

    def __getitem__(self, index):
        """
            Return : PIL image, sparse label
        """
        x = Image.open(self.data[index]).convert('RGB')
        x = x.resize(self.target_size[:2],Image.Resampling.BILINEAR)
        

        return x, self.label[index]

    def __len__(self):
        return len(self.data)

class FaceClfDataset(ImageDataset):
    def __init__(self, root, classes, target_size: tuple, shuffle=False, transform=None, class_mode=None, **kwargs):
        super().__init__(root, classes, target_size, shuffle, **kwargs)
        # transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.allow_class_modes =  ["categorical","sparse",None]

        if class_mode is not None:
            self.class_mode = class_mode
        else:
            self.class_mode = self.allow_class_modes[0]

    def _class_mode(self,y):
        if self.class_mode == self.allow_class_modes[0]:
            return self._to_categorical_class(y)
        elif self.class_mode == self.allow_class_modes[1]:
            return torch.tensor(y,dtype=torch.long)

    def _to_categorical_class(self,y):
        temp = torch.zeros((len(self.classes),),dtype=torch.long)
        temp[y] = 1
        return temp 

    def categorical2sparse(y:torch.Tensor):
        return torch.argmax(y,1)

    def __getitem__(self, index):
        x,y = super().__getitem__(index)
        return self.transform(x),self._class_mode(y)

    def get_sample(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()    

def get_train_dataset(imgs_folder):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1 # tuple[tensor, label] label + 1
    return ds, class_num


def split_dataset(dataset,test_size,shuffle=False):
    # dataset class only have this attribute
    datas = dataset.data
    labels = dataset.label

    train_proportion = 1. - test_size
    test_proportion = test_size

    train_indices = np.array([], dtype=np.int64)
    test_indices = np.array([], dtype=np.int64)

    for c in np.unique(labels):
        indices_c = np.where(labels == c)[0]
        np.random.shuffle(indices_c)
        num_train = int(np.round(len(indices_c) * train_proportion))
        num_test = len(indices_c) - num_train
        train_indices_c = indices_c[:num_train]
        test_indices_c = indices_c[num_train:num_train+num_test]
        train_indices = np.concatenate((train_indices, train_indices_c))
        test_indices = np.concatenate((test_indices, test_indices_c))
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    return Subset(dataset,train_indices), Subset(dataset,test_indices)