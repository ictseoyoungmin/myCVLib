import torch
import torch.nn as nn
import numpy as np
from torchvision import utils
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
        if type(tensor) == np.ndarray:
                tensor = torch.from_numpy(tensor)
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

def visNumpy(ndarray:np.ndarray, ch=0, allkernels=False, nrow=8, padding=1): 
        n,c,w,h = ndarray.shape

        if allkernels: ndarray = ndarray.reshape(n*c, -1, w, h)
        elif c != 3: ndarray = ndarray[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((ndarray.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(ndarray, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))


# reference : https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
class VerboseExecution(nn.Module):
    """
    example : verbose_model = VerboseExecution(pfld_backbone)
              _ = verbose_model(transform(img1).unsqueeze(0))
    """
    def __init__(self, model: nn.Module,permute = True):
        super().__init__()
        self.model = model
        self.layer_output = []

        def trans(tensor:torch.Tensor):
            try:
                return tensor.permute(0,2,3,1).data.numpy()
            except:
                return tensor.data.numpy()
        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )
            layer.register_forward_hook(
                lambda layer, input, output: self.layer_output.append([layer.__name__,(trans(output) if permute else output)])
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model(x)
        return x

def plot_feature_map(instance,col=8,fig_size=(12,10),use_title=False):
    num_features =instance.shape[-1]
    row = int(num_features / col) +1

    fig = plt.figure(figsize=fig_size)
    for i in range(1,num_features+1):
        ax = fig.add_subplot(row,col,i)
        ax.imshow(instance[:,:,i-1])
        ax.axis(False)
    if use_title:
        fig.suptitle(f"{num_features} features in current layer",fontsize=42)
        fig.tight_layout()

def all_plot_feature(dic:dict,use_title=False,*args):
    used_key = []
    for key in dic.keys():
        if ('conv' in key and 'flatten' not in key) or ('block' in key):
            plot_feature_map(dic[key][0],10,use_title=use_title)
            used_key.append(key)
    
    for arg in args:
        try:
            if (str(arg) in dic.keys()) and (arg not in used_key):
                plot_feature_map(dic[key][0],10,use_title=use_title)
        except:
            continue

