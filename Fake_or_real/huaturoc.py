from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

y_label=np.load('./target.npy')
y_pre = np.load('./predict.npy')


classname=['Fake face','Real face']
sum=0
_,axe=plt.subplots(figsize=(6,6))
fpr1=[]
tpr1=[]
thersholds1=[]
for i in range(2):
    fpr, tpr, thersholds = roc_curve(y_label[:,i], y_pre[:,i], pos_label=1)
    fpr1.append(fpr)
    tpr1.append(tpr)
    thersholds1.append(thersholds)

    # for i, value in enumerate(thersholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)
    sum=roc_auc+sum
    plt.plot(fpr, tpr, label='%s (AUC = %.4f)'%(classname[i],roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR',weight='bold',fontsize=10)
    plt.ylabel('TPR',weight='bold',fontsize=10)  # 可以使用中文，但需要导入一些库即字体

# print(fpr11)
# roc_auc = auc(fpr11, tpr11)
newfpr=np.zeros(100)
newtpr=np.zeros(100)
for k in range(2):
    length=len(fpr1[k])
    newfpr1=np.zeros(100)
    newtpr1 = np.zeros(100)
    step=length/100
    for i in range(100):
        newfpr1[i]=np.mean(fpr1[k][int(i*step):int((i+1)*step)])
        newtpr1[i] = np.mean(tpr1[k][int(i * step):int((i + 1) * step)])
    newfpr=newfpr+newfpr1
    newtpr = newtpr + newtpr1

plt.plot(newfpr/2, newtpr/2, '--',label='%s (AUC = %.4f)'%('average curve',sum/2), lw=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
# axe.spines['right'].set_visible(False)
# axe.spines['top'].set_visible(False)
plt.title('FFHQ ROC curve',weight='bold',fontsize=10)
plt.legend(loc="lower right",fontsize=10)
# axe.grid(axis='y',c='silver',ls='--',zorder=-1)
# axe.grid(axis='x',c='silver',ls='--',zorder=-1)
plt.tight_layout()
plt.savefig('./image/dataset.png',dpi=300)
plt.show()
