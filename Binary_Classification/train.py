import torch
import time
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report,accuracy_score, confusion_matrix
import seaborn as sns

# def train(model, device, num_epoch, dataset,loss_function, optimizer,threshold, save_path):
def train(model, device, num_epoch, dataset,loss_function, optimizer, scheduler,threshold, save_path):
    #학습 저장할 변수들
    best_model_wts = copy.deepcopy(model.state_dict()) # 초기 모델 속 파라미터 저장해둠
    best_loss = 10000 #첫 loss를 best loss로 두고 학습할 수 있도록 큰 사이즈 지정

    epochs = []#loss, epoch 그래프 그리기 위해 epoch 저장하기

    #그래프 그리기 위해 train과 valid accuracy, loss 저장
    accuracy = {'train':[], 'valid':[]}
    losses = {'train':[], 'valid':[]}
    targets = {'train':[], 'valid':[]}
    preds = {'train':[], 'valid':[]}

    for epoch in range(num_epoch): #데이터 셋 한 번 보고 학습함 = 1 epoch
        print(f"{epoch+1}/{num_epoch}")
        running_loss = [] # loss 저장해줄 변수
        running_acc = []
        epochs.append(epoch)
        for phase in ['train', 'valid']: #train 한 번 할 때마다 valid도 한 번 해줌
            

            #train할 땐 학습하고 valid 시 eval로 모델 평가함
            if phase=='train':
                model.train()
                
            else:
                model.eval()
            
            for image, label in dataset[phase]:
                optimizer.zero_grad() #backpropagation전에 gradients 값 zero 만들어주고 시작하기
                #batch 사이즈 별로 묶어준 데이터들 학습시작
                #image와 label 학습 위해 GPU에 올려둠
                image = image.to(device)
                label = label.to(device)
                
                if phase=='train':
                    #모델에게 사진 보여주고 출력값 받기
                    with torch.set_grad_enabled(True): #train 시에는 gradient 업로드해줌
                        outputs = model(image)
                        loss = loss_function(outputs.to(torch.float32),label.to(torch.float32))
                        pred = [1 if output>=threshold else 0 for output in outputs]
                        targets[phase] += np.array(label.cpu()).flatten().tolist() #1차원으로 쭉 넣어주기 위해 append하지 않고 더해줌
                        preds[phase] += pred
                        
                        #학습 파라미터 업데이트
                        loss.backward()
                        optimizer.step()

                else:
                    with torch.no_grad(): # valid는 평가하는 것이기 때문에 학습 gradient 업데이트를 꺼줌
                        outputs = model(image)
                        loss = loss_function(outputs.to(torch.float32),label.to(torch.float32))
                        pred = [1 if output>=threshold else 0 for output in outputs]
                        targets[phase] += np.array(label.cpu()).flatten().tolist() #1차원으로 쭉 넣어주기 위해 append하지 않고 더해줌
                        preds[phase] += pred

                        if best_loss > loss:#loss가 최소가 되는 모델을 best로 선정하여 저장하기
                            best_loss = loss
                            best_model_wts = copy.deepcopy(model.state_dict())
                            torch.save(model.state_dict(best_model_wts), f"{save_path}/weight/best_weight.pt")

                acc = accuracy_score(np.array(label.cpu()).flatten().tolist(), pred)
                running_loss.append(loss)
                running_acc.append(acc)
                scheduler.step()
            #1epoch 끝나는 시점. 학습 결과들 정리 후 출력해주기
            loss = sum(running_loss)/len(running_loss)
            losses[phase].append(loss.item())

            accs = sum(running_acc)/len(running_acc)
            accuracy[phase].append(accs)
            print("%s | loss %.4f accuracy %.4f"%(phase, loss, accs))
    
    #학습 끝난 지점 학습동안 모았던 결과들 출력하기
    plt.plot(epochs, losses['train'], label='train loss', color='red')
    plt.plot(epochs, losses['valid'], label='valid loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"{save_path}/plot/loss_graph.png")
    plt.close()

    
    plt.plot(epochs, accuracy['train'], label='train accuracy', color='red')
    plt.plot(epochs, accuracy['valid'], label='valid accuracy', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"{save_path}/plot/accuray_graph.png")
    plt.close()   

    model.load_state_dict(best_model_wts)

    for phase in ['train', 'valid']:
        #auc, f1, precision 등 학습 지표
        auc = roc_auc_score(targets[phase],preds[phase])
        print("%s | AUC : %.4f"%(phase, auc))
        print(
            classification_report(targets[phase],preds[phase])
        )
        targets = []
        preds = []
        for image, label in dataset[phase]:
            image = image.to(device)
            outputs = model(image)
            preds = [1 if output>=threshold else 0 for output in outputs]

            targets += np.array(label.cpu()).flatten().tolist() #1차원으로 쭉 넣어주기 위해 append하지 않고 더해줌
            preds += pred
        
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.xlabel('Prediction label')
        plt.ylabel("True label")
        plt.xticks([0.5,1.5],['0(B)', '1(M)'])
        plt.yticks([0.5,1.5],['0(B)', '1(M)'])
        plt.savefig(f"{save_path}/plot/{phase}_confusion_map.png")
        plt.close()

def inference(model, device, image):
    #test 함수
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
    
    return output