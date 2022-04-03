from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
def confusion_matrix_pyplot(y_true,y_pred,num_classes,name=""):
       if isinstance(y_true,torch.Tensor):
              y_true=y_true.clone().detach().cpu().numpy()
       if isinstance(y_pred,torch.Tensor):
              y_pred=y_pred.clone().detach().cpu().numpy()
       if y_true.shape[-1]==num_classes:
              y_true=np.argmax(y_true,1)
       if y_pred.shape[-1]==num_classes:
              y_pred=np.argmax(y_pred,1)
       array=confusion_matrix(y_true,y_pred)
       sns.set(font_scale=2.0)
       df=pd.DataFrame(array,index=range(num_classes),columns=range(num_classes))
       fig=plt.figure(figsize=(10,10))
       ax=fig.add_subplot(111)
       ax=sns.heatmap(df,square=True,annot=True,ax=ax,cmap="YlGnBu")
       plt.title("---")
       fig=ax.get_figure()
       if name!="":
              fig.savefig(name)
       else:
              fig.savefig("test.png")