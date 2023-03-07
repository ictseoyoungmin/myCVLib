import os,sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import PIL.Image as Image


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues,cm_proba=True):
    if cm_proba:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

def plot_confusion_matrix_detail(cm,classes):
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plot_confusion_matrix(cm,classes)
    plt.subplot(122)
    plot_confusion_matrix(cm,classes,cm_proba=False)
    
def sample_from_derectory(base_path=None,img_shape=(224,224),labels=None): # func only this file
    f, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize = (16,7))
    samples = []
    for ax in axes.ravel():
        label = np.random.choice(os.listdir(base_path))
        i = np.where(np.array(labels)==label)[0]
        img = np.random.choice(os.listdir(os.path.join(base_path, label)))
        img = Image.open(os.path.join(base_path, label) + "/" + img) # os.path.join
        img = img.resize(img_shape[:2],resample=Image.Resampling.NEAREST)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label+" "+str(i))  
        samples.append(np.array(img))
    
    return np.array(samples)

def plot_result(samples,preds,true_label=None,labels=[None,None]):
    f, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize = (16,7))
    
    for i,ax in enumerate(axes.ravel()):
        label = labels[preds[i]]
        l = preds[i]
        img = Image.fromarray(samples[i])
        ax.imshow(img)
        ax.axis('off')
        true_text = f"T : {true_label}" if true_label is not None else ""
        ax.set_title("P : "+label+" "+str(l) + true_text)  

def ipy_path_append(root=None):
    r = root if root is not None else os.getcwd()
    for path in os.listdir(r):
        if os.path.isdir(path):
            ipy_path_append(path)
            if path not in sys.path:
                sys.path.append(path)


class ConfusionMatrix:
  def __init__(self,classes = None):
    self.cm = None
   
    if classes is not None:
      self.classes = classes
      self.n_classes = len(classes)

  def get_conf_matrix(self,actual,pred):
    if self.classes is None:
      self.classes = np.unique(actual)
      self.n_classes = len(self.classes)
    conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int32)
    for i in range(self.n_classes):
        for j in range(self.n_classes):
            conf_matrix[i, j] = np.sum((actual == self.classes[i]) & (pred == self.classes[j]))

    return conf_matrix

  def update(self,actual,pred):

    curr_cm = self.get_conf_matrix(actual,pred)
    if self.cm is not None:
      self.cm += curr_cm
    else:
      self.cm = curr_cm

  def get_result(self):
    return self.cm










