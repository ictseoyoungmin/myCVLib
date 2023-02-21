import torch
import numpy as np

class ConfusionMatrix:
    """
    example : ConfusionMatrix([0,1,2])
    """
    def __init__(self,classes = None):
        self.cm = None

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

    def get_conf_matrix(self,actual,pred):
        if self.classes is None:
            self.classes = torch.unique(actual)
            self.n_classes = len(self.classes)
        conf_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int32)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                conf_matrix[i, j] = torch.sum((actual == self.classes[i]) & (pred == self.classes[j]))

        return conf_matrix

    def update(self,actual,pred):

        curr_cm = self.get_conf_matrix(actual,pred)
        if self.cm is not None:
            self.cm += curr_cm
        else:
            self.cm = curr_cm

    def get_acc(self):
        """
        same return np.sum(np.diag(cm))/np.sum(np.sum(cm))
        """
        _sum = 0
        tp = 0
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                _sum += self.cm[i,j]
                if i == j:
                    tp += self.cm[i,j]
        return tp / _sum
        
    def get_result(self):
        return self.cm
    
def classfication_report(cm,keys = ['unknown','cat','dog']):
    assert  type(cm) is not list , Exception('keys type error, not support list type')
    if type(cm) is torch.Tensor:
        cm = cm.numpy()

    print(np.sum(np.sum(cm)))
    targets = len(keys)
    margin = 3
    first_width = max(9,*[len(k) for k in keys])
    
    # todo: no yame
    margins = [first_width+margin+len('precision'),len('recall')+margin,len('f1-score')+margin]
    print(f"{'precision':>{margins[0]+1}} {'recall':>{margins[1]+3}} {'f1-score':>{margins[2]+3}}")
    margins = [m - first_width for m in margins]
    
    # macro
    macro_pre = 0.
    macro_re = 0.
    macro_f1 = 0.
    for target in range(targets):
        precision = cm[target,target] / sum(cm[:,target])
        recall = cm[target,target] / sum(cm[target,:])
        f1 = 2*(1/(1/recall + 1/precision))
        print(f"{keys[target]:>{first_width}s} {precision:>{margins[0]}.4f} {recall:>{sum(margins[:2])}.4f} {f1:>{sum(margins[:])}.4f}")
        macro_pre+=precision
        macro_re+=recall
        macro_f1+=f1
    print()
    print(f"{'macro avg':>{first_width+margins[1]}s} {macro_pre/targets:>{margins[0]}.4f} {macro_re/targets:>{sum(margins[:2])}.4f} {macro_f1/targets:>{sum(margins[:])}.4f}")
    
    # micro
    TP = sum(np.diag(cm))
    FP = np.sum(cm, axis=0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    acc = TP/sum(sum(cm))
    micro_pre = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    micro_re = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    micro_f1 = 2 * micro_pre * micro_re / (micro_pre + micro_re)
    
    print(f"{'micro avg':>{first_width+margins[1]}s} {micro_pre:>{margins[0]}.4f} {micro_re:>{sum(margins[:2])}.4f} {micro_f1:>{sum(margins[:])}.4f}")
    print(f"{'accuarcy':>{first_width+margins[1]}s} {'':>{margins[0]}s} {'':>{sum(margins[:2])}s} {acc:>{sum(margins[:])}.4f}")
        # todo : support pass

