import torch
import time
import copy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from PIL import Image
def train_model(model, dataloader, loss_function, optimizer, num_epochs,device):
    since = time.time()
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    train_target = []
    valid_target = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

                for _, (image, label) in dataloader[phase]:
                    image = Image.open(image).to(device)
                    label = label.to(device)

                    
                    with torch.set_grad_enabled(phase=='train'):
                        #학습
                        pred = model(image)
                        loss = loss_function(pred, label)
                        outputs = torch.argmax(pred, 1)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        #학습 지표들 정리를 위한 변수 정리
                        train_loss.append(loss)
                        train_acc.append((outputs==label).sum().item())
                        train_target.append(label)
                    
                    if phase=='valid':
                        #평가
                        with torch.no_grad():
                            pred = model(image)
                            loss = loss_function(pred, label)
                            outputs = torch.argmax(pred, 1)

                            valid_loss.append(loss)
                            valid_acc.append((outputs==label).sum().item())
                            valid_target.append(label)

                        if valid_loss[-1] < best_loss:
                            best_loss = valid_loss[-1]
                            best_model_wts = copy.deepcopy(model.state_dict())

                    
                if phase=='train':
                    acc = 100*float(sum(train_acc))/len(dataloader['train'])
                    print(f"train loss:{train_loss[-1]} train accuracy:{acc}")
                else:
                    acc = 100*float(sum(valid_acc))/len(dataloader['valid'])
                    print(f"valid loss:{valid_loss[-1]} valid accuracy:{acc}")
    #학습 지표 정리하기
    train_f1_score = f1_score()
    
                