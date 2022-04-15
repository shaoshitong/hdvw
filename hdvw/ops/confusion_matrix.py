from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def confusion_matrix_pyplot(y_true,y_pred,num_classes=None,name=""):
    """
    y_true: true label
    y_pred: model output
    num_classes: the num classes of dataset
    """
    if isinstance(y_true,torch.Tensor):
        y_true=y_true.clone().detach().cpu().numpy().astpye(float)
    elif isinstance(y_true,np.ndarray):
        y_true=y_true.astype(float)
    else:
        raise NotImplementedError(f"this type {type(y_true)} is not allowed")
    if isinstance(y_pred,torch.Tensor):
        y_pred=y_pred.clone().detach().cpu().numpy()
    elif isinstance(y_true,np.ndarray):
        y_pred=y_pred.astype(float)
    else:
        raise NotImplementedError(f"this type {type(y_true)} is not allowed")
    if num_classes!=None:
        if y_true.ndim==2 and y_true.shape[-1]==num_classes:
              y_true=np.argmax(y_true,1)
        if y_pred.ndim==2 and y_pred.shape[-1]==num_classes:
              y_pred=np.argmax(y_pred,1)
    else:
        import warnings
        warnings.warn("may be error")
    print(f"ture label shape is {y_true.shape}, pred label shape is {y_pred.shape}")
    array=confusion_matrix(y_true,y_pred)
    sns.set(font_scale=2.0)
    df=pd.DataFrame(array,index=range(num_classes),columns=range(num_classes))
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    ax=sns.heatmap(df,square=True,annot=True,ax=ax,cmap="YlGnBu")
    plt.title("confusion matrix visualization")
    fig=ax.get_figure()
    if name!="":
          fig.savefig(name)
    else:
          fig.savefig("test.png")
