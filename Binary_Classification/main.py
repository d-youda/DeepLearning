from data.dataloader import Dataloader
from torch.utils.data import DataLoader
import torch
from densenet import Model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from train import train, inference
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, classification_report,accuracy_score, confusion_matrix
import numpy as np
torch.cuda.empty_cache()
#학습 파라미터
epochs = 100
save_path = 'D:\\DeepLearning\\Binary_Classification\\result'
data_path = 'D:\\DeepLearning\\Binary_Classification\\dataset'
batch_size = 64
threshold = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

train_dataload = Dataloader(path=data_path, mode=f'train')
train_dataset = DataLoader(dataset=train_dataload,batch_size=batch_size, shuffle=False,pin_memory=True)

valid_dataload = Dataloader(path=data_path, mode=f'valid')
valid_dataset = DataLoader(dataset=valid_dataload,batch_size=batch_size, shuffle=False,pin_memory=True)

test_dataload = Dataloader(path=data_path, mode=f'test')
test_dataset = DataLoader(dataset=test_dataload,batch_size=1, shuffle=False,pin_memory=True)

data = {'train': train_dataset, 'valid':valid_dataset}
dataset_sizes = {'train': len(train_dataload), 'valid':len(valid_dataload), 'test':len(test_dataload)}
print(dataset_sizes)

model = Model()
model.to(device)

loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.RMSprop(model.parameters(),lr=0.00001)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

# 학습 진행하기
# model = train(model,device, epochs, data, loss_function, optimizer, threshold, save_path)
model = train(model,device, epochs, data, loss_function, optimizer,scheduler, threshold, save_path)
model.load_state_dict(torch.load(f'{save_path}/weight/best_weight.pth'))

targets = []
preds = []
for image_num, (image, label) in enumerate(test_dataset):
    #test위해 image GPU에 올리기
    image = image.to(device)

    output = inference(image)
    pred = 1 if output>= threshold else 0

    targets.append(label)
    preds.append(pred)

print("test | accuracy %.4f AUC %.4f "%(accuracy_score(targets, preds), roc_auc_score(targets, preds)))
print(
    classification_report(targets, preds)
)

#confusion matrix 그리기
cm = confusion_matrix(targets, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Prediction label')
plt.ylabel("True label")
plt.xticks([0.5,1.5],['0(B)', '1(M)'])
plt.yticks([0.5,1.5],['0(B)', '1(M)'])
plt.savefig(f"{save_path}/plot/test_confusion_map.png")