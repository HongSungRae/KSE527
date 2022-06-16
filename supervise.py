from torch.utils.data import DataLoader
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import time
import json
from sklearn import metrics

# local
from utils import adjust_learning_rate, write_log, draw_curve, get_confusion_matrix
from dataset import FMA
from models.simsiam import Evaluator, Siamusic
from parameters import *
from metrics import get_recall_precision

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--size', default='small', type=str, choices=['small', 'medium'])
parser.add_argument('--optim', default='SGD', type=str, help='SGD or Adam')
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--backbone', default='resnet50', type=str, help='Select the model among resnet50, resnet101, resnet152')
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--pred_dim', default=512, type=int)
parser.add_argument('--sr', default=22050, type=int)
parser.add_argument('--n_fft', default=512, type=int)
parser.add_argument('--f_min', default=0.0, type=float)
parser.add_argument('--f_max', default=8000.0, type=float)
parser.add_argument('--n_mels', default=80, type=int)


parser.add_argument('--gpu_id', default='1', type=str)

args = parser.parse_args()



def train(train_loader, model, criterion, optimizer):
    model.train()
    train_epoch_loss = 0
    

    for i,(audio, label) in enumerate(train_loader):
        audio = audio.type(torch.float32).cuda()
        label = label.type(torch.LongTensor).cuda()
        pred = model(audio)

        optimizer.zero_grad()
        train_loss = criterion.cuda()(pred, label)
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()

        if i%10 == 0:
            print(f'[ {i} | {len(train_loader)} ] Train_loss : {np.around(train_loss.item(), 3)}')
    return train_epoch_loss



def validation(valid_loader, model, criterion):
    model.eval()
    valid_epoch_loss = 0

    with torch.no_grad():
        for i, (audio, label) in enumerate(valid_loader):
            audio = audio.type(torch.float32).cuda()
            label = label.type(torch.LongTensor).cuda()
            pred = model(audio)
            valid_loss = criterion.cuda()(pred, label)

            valid_epoch_loss += valid_loss.item()
    
    return valid_epoch_loss





def test(test_loader, model, save_path):
    print('='*10 + ' Transfer Test ' + '='*10)
    
    model.eval()
    pred_list = []
    label_list = []
    
    with torch.no_grad():
        for audio, label in tqdm(test_loader):
            label_list += label.tolist()
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
    
            pred = model(audio)
            pred = torch.argmax(torch.sigmoid(pred),dim=-1)
            pred_list += pred.tolist()
            
    confusion_matrix = get_confusion_matrix(pred_list,label_list,8)
    recall_at_1, precision_at_1 = get_recall_precision(confusion_matrix)
    acc_at_1 = float(metrics.accuracy_score(label_list,pred_list))
    f1_at_1 = 2*recall_at_1*precision_at_1/(recall_at_1+precision_at_1+1e-3)

    result = {
              'Acc' : f'{acc_at_1:.3f}', 
              'Precision@1' : f'{precision_at_1:.3f}',
              'Recall@1' : f'{recall_at_1:.3f}',
              'F1@1' : f'{f1_at_1:.3f}'
              }

    # Save result, confusion matrix
    with open(f'{save_path}/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    f = open(f'{save_path}/confusion.txt','w')
    f.write(str(confusion_matrix))

    print(result)
    print(f'Confusion Matrix : \n{confusion_matrix}')
    print("=================== Test End ====================")



def main():
    ####### Save Path #######
    current_time = str(time.time()).split('.')[-1]
    save_path = './exp/' + current_time

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ####### Configuration #######
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ####### GPU enviorement ######
    os.environ['CUDA_LAUNCH_BLOCKING'] = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ####### dataloader setting #######
    train_data = FMA(split='training',size=args.size)
    valid_data = FMA(split='validation',size=args.size)
    test_data = FMA(split='test',size=args.size)

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    ####### model setting #######
    model = Siamusic(backbone=args.backbone,
                     augmentation='basic', 
                     dim=args.dim, 
                     pred_dim=args.pred_dim, 
                     sr=args.sr, 
                     n_fft=args.n_fft, 
                     f_min=args.f_min, 
                     f_max=args.f_max, 
                     n_mels=args.n_mels)
    supervised_model = Evaluator(encoder=model.encoder,
                                 num_classes=8,
                                 augmentation='basic',
                                 dim=args.dim,
                                 sample_rate=22050, 
                                 n_fft=args.n_fft, 
                                 f_min=args.f_min, 
                                 f_max=args.f_max, 
                                 n_mels=args.n_mels).cuda()
    
   

    ####### loss setting #######
    criterion = nn.CrossEntropyLoss()
    

    ####### optimizer setting #######
    init_lr = args.lr * args.batch_size / 256
    
    optim_params = supervised_model.parameters()

    if args.optim == 'SGD':
        optimizer = optim.SGD(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ####### training ########
    best_loss = 10000.0
    best_epoch = 0
    count = 0
    train_loss_dic = {}
    valid_loss_dic = {}

    
    for epoch in tqdm(range(args.num_epochs)):
        print(' ')
        start_time = time.time()
        
        print('='*10 + f' Training [{epoch+1}/{args.num_epochs} | eph/ephs] ' + '='*10)
        train_epoch_loss = train(train_loader, supervised_model, criterion, optimizer)
        train_epoch_loss = train_epoch_loss/len(train_loader)

        valid_epoch_loss = validation(valid_loader, supervised_model, criterion)
        valid_epoch_loss = valid_epoch_loss/len(valid_loader)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        write_log(save_path, 'train_log', str(epoch+1), np.around(train_epoch_loss, 3))
        write_log(save_path, 'valid_log', str(epoch+1), np.around(valid_epoch_loss, 3))
        train_loss_dic[epoch+1] = train_epoch_loss
        valid_loss_dic[epoch+1] = valid_epoch_loss

        print(f'========== Train CE Loss: {train_epoch_loss:.5f} | Valid CE Loss " {valid_epoch_loss:.5f} | It took {(time.time()-start_time)/60:.3f} mins ==========')
        
        
        if valid_epoch_loss > best_loss:
            count += 1
            if (epoch>30) and (count>=5):
                print('='*20)
                print(f'Early Stopping At {epoch+1} epoch : The model saved at {best_epoch+1}epoch lastly!')
                break
            else:
                pass
        elif valid_epoch_loss <= best_loss:
            count = 0
            best_loss = valid_epoch_loss
            best_epoch = epoch
            # torch.save(train_model.state_dict(), f'{save_path}/pre.pth')
            torch.save(supervised_model, f'{save_path}/supervised.pth')
            print('='*20)
            print(f'Saved At {epoch+1} epoch : Model saved!')

    del(supervised_model)
    supervised_model = torch.load(f'{save_path}/supervised.pth')
    supervised_model.cuda()
    test(test_loader, supervised_model,save_path)
    draw_curve(save_path, train_loss_dic, 'train_loss')
    draw_curve(save_path, valid_loss_dic, 'valid_loss')

if __name__ == '__main__':
    main()