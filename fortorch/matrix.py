import torch

class ConfusionMatrix:
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

  def get_result(self):
    return self.cm


