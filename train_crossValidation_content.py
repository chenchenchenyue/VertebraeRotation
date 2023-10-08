import math
from nis import match
from time import time
from tqdm import tqdm
import torchvision.transforms as transforms
from network_all import MLP
import torch
from tensorboardX import SummaryWriter
import argparse
from dataset import MedicalDataset
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

###计算loss
def mlt(y_pred,y_true,log_vars):
    ys_true = y_pred
    ys_pred = y_true
    loss=0
    for y_true, y_pred, log_var in zip(ys_true, ys_pred, log_vars):
        pre = torch.exp(-log_var)
        loss += torch.sum(pre*(y_true-y_pred)**2+log_var,-1)
    loss = torch.mean(loss)
    return loss
###计算向量loss
def DIYloss(y_preds,y_trues):
    loss=0
    for y_pred, y_true in zip(y_preds,y_trues):

        dotVal = torch.dot(y_true,y_pred)
        fenMother = torch.linalg.norm(y_true)*torch.linalg.norm(y_pred)
        returnVal = dotVal/fenMother
        loss += 1-returnVal
    return loss/y_preds.size()[0]
###计算三个向量夹角的loss
def DIYlossadd(outputs1,outputs2,outputs3):
    ####其次三个向量相互垂直
    loss = 0
    for output1,output2,output3 in zip(outputs1,outputs2,outputs3):
        value1 = torch.dot(output1,output2)
        value2 = torch.dot(output2,output3)
        value3 = torch.dot(output1,output3)
        loss += torch.pow((value1),2)+torch.pow((value2),2)+torch.pow((value3),2)
    return loss/outputs1.size()[0]



def mainTrain(foldnum=1):
    #####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type = tuple, default = (1024, 1024))
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--data_dir', type = str, default = 'cocodata')
    parser.add_argument('--data_set', type = str, default = 'train')

    parser.add_argument('--num_classes', type = int, default = 2)
    parser.add_argument('--device', type = str, default = "cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_epoch', type = int, default = 200)
    parser.add_argument('--model_save_path', type = str, default = 'checkpoint')

    args = parser.parse_args()
    


    cudax = 2
    max_float=float('inf')  ##初始为最大loss
    writer_train = SummaryWriter('./runs/train_fold{}'.format(foldnum))
    writer_test = SummaryWriter('./runs/test_fold{}'.format(foldnum))
    path = "./new裁剪图片/"

    train_loader = DataLoader(MedicalDataset(foldnum,"train",args), batch_size = args.batch_size, shuffle = True,
                                       collate_fn = lambda x: tuple(zip(*x))
                                       )
    valiation_dl = DataLoader(MedicalDataset(foldnum,"val",args), batch_size = args.batch_size, shuffle = True,
                                       collate_fn = lambda x: tuple(zip(*x))
                                       )



    #####################################定义网络
    model = MLP()
    model.cuda(cudax)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # 优化函数 , betas=(0.9, 0.999), weight_decay=1e-3
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print('---------- Networks initialized -------------')
    # summary(model, (1, 128, 128))
    print('-----------------------------------------------')
    trainIndex = 0
    testIndex = 0
    start = time()
    losses = []
    min_loss = 100000
    for epoch in tqdm(range(args.max_epoch)):
        print("epoch:{}#############################################".format(epoch+1))
        train_loss = 0.0
        valid_loss = 0.0
        #######################################################
        # Training Data
        #######################################################
        model.train()
        for step, sample in enumerate(train_loader):
      
            image_F1 = torch.stack(sample[0]).cuda(cudax)
            target_F1 = sample[1]
            ori_image_F1 = sample[2]
            image_F2 = torch.stack(sample[3]).cuda(cudax)
            target_F2 = sample[4]
            ori_image_F2 = sample[5]
            ######
            targetL3_1 = sample[6]
            targetL3_2 = sample[7]
            targetL3_3 = sample[8]
            targetL3_1 =torch.tensor([tuple(eval(target)) for target in targetL3_1]).cuda(cudax)
            targetL3_2 = torch.tensor([tuple(eval(target)) for target in targetL3_2]).cuda(cudax)
            targetL3_3 = torch.tensor([tuple(eval(target)) for target in targetL3_3]).cuda(cudax)
            ######
            targetL4_1 = sample[9]
            targetL4_2 = sample[10]
            targetL4_3 = sample[11]
            targetL4_1 =torch.tensor([tuple(eval(target)) for target in targetL4_1]).cuda(cudax)
            targetL4_2 = torch.tensor([tuple(eval(target)) for target in targetL4_2]).cuda(cudax)
            targetL4_3 = torch.tensor([tuple(eval(target)) for target in targetL4_3]).cuda(cudax)
            ######
            targetL5_1 = sample[12]
            targetL5_2 = sample[13]
            targetL5_3 = sample[14]
            targetL5_1 =torch.tensor([tuple(eval(target)) for target in targetL5_1]).cuda(cudax)
            targetL5_2 = torch.tensor([tuple(eval(target)) for target in targetL5_2]).cuda(cudax)
            targetL5_3 = torch.tensor([tuple(eval(target)) for target in targetL5_3]).cuda(cudax)

            imagename = sample[15]
            
            image_F1_list = list(image.to(args.device) for image in image_F1)
            targets1 = [{k : v.to(args.device) for k, v in t.items()} for t in target_F1]
            image_F2_list = list(image.to(args.device) for image in image_F2)
            targets2 = [{k : v.to(args.device) for k, v in t.items()} for t in target_F2]
            
 
            optimizer.zero_grad()
            #########
            # outputs = model(f1data,f2data)
        
            lossoutput1,lossoutput2,outputs11,outputs12,outputs13,outputs21,outputs22,outputs23,outputs31,outputs32,outputs33=model(image_F1, image_F2,image_F1_list, image_F2_list,targets1,targets2,typemode="train")
            lossmask1 = sum(loss for loss in lossoutput1.values())
            lossmask2 = sum(loss for loss in lossoutput2.values())
            loss11 = DIYloss(outputs11,targetL3_1)
            loss12 = DIYloss(outputs12,targetL3_2)
            loss13 = DIYloss(outputs13,targetL3_3)
            
            loss21 = DIYloss(outputs21,targetL4_1)
            loss22 = DIYloss(outputs22,targetL4_2)
            loss23 = DIYloss(outputs23,targetL4_3)

            loss31 = DIYloss(outputs31,targetL5_1)
            loss32 = DIYloss(outputs32,targetL5_2)
            loss33 = DIYloss(outputs33,targetL5_3)

            loss = lossmask1+lossmask2+loss11+loss12+loss13+loss21+loss22+loss23+loss31+loss32+loss33

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'[{epoch + 1:02d}/{args.max_epoch:02d}], train_loss:{train_loss / len(train_loader):.5f}')
        losses.append(round(float(train_loss) / len(train_loader), 5))
            
            # 保存 fine-tuning 后的模型
        model_save_path = Path(args.model_save_path)
        if not model_save_path.exists() :
                model_save_path.mkdir(parents = True, exist_ok = True)

        if train_loss < min_loss :
                min_loss = train_loss
                torch.save(model.state_dict(), model_save_path.joinpath(f'mask_rcnn_with_fold_{foldnum}-epoch.pth'))

            
        #     loss_add = DIYlossadd(outputs1,outputs2,outputs3)
        #     ###########
        #     loss = loss1+loss2+loss3+0.1*loss_add
        #     # loss = loss1+loss2+loss3
        #     loss.backward()
        #     optimizer.step()
        #     train_loss += loss.item()
        #     stepLR.step()
        #     if (step+1)%3 == 0:
        #         writer_train.add_scalars('train', {
        #                 'train_loss': train_loss /(step+1),
        #                 'train_loss_val1': loss1.item(),
        #                 'train_loss_val2': loss2.item(),
        #                 'train_loss_val3': loss3.item(),
        #             }, trainIndex)
        #         trainIndex += 1
        # print('epoch:[%d] train_loss: %.3f' % (epoch + 1,  train_loss /(step+1)))
        # #######################################################
        # #Validation Step
        # #######################################################
        # model.eval()
        # with torch.no_grad():  # to increase the validation process uses less memory
        #     for step, sample in enumerate(valiation_dl):
        #         image_F1 = sample[0]
        #         target_F1 = sample[1]
        #         ori_image_F1 = sample[2]
        #         image_F2 = sample[3]
        #         target_F2 = sample[4]
        #         ori_image_F2 = sample[5]
        #         ######
        #         targetL3_1 = sample[6]
        #         targetL3_2 = sample[7]
        #         targetL3_3 = sample[8]
        #         targetL3_1 =torch.tensor([tuple(eval(target)) for target in targetL3_1]).cuda(cudax)
        #         targetL3_2 = torch.tensor([tuple(eval(target)) for target in targetL3_2]).cuda(cudax)
        #         targetL3_3 = torch.tensor([tuple(eval(target)) for target in targetL3_3]).cuda(cudax)
        #         ######
        #         targetL4_1 = sample[9]
        #         targetL4_2 = sample[10]
        #         targetL4_3 = sample[11]
        #         targetL4_1 =torch.tensor([tuple(eval(target)) for target in targetL4_1]).cuda(cudax)
        #         targetL4_2 = torch.tensor([tuple(eval(target)) for target in targetL4_2]).cuda(cudax)
        #         targetL4_3 = torch.tensor([tuple(eval(target)) for target in targetL4_3]).cuda(cudax)
        #         ######
        #         targetL5_1 = sample[12]
        #         targetL5_2 = sample[13]
        #         targetL5_3 = sample[14]
        #         targetL5_1 =torch.tensor([tuple(eval(target)) for target in targetL5_1]).cuda(cudax)
        #         targetL5_2 = torch.tensor([tuple(eval(target)) for target in targetL5_2]).cuda(cudax)
        #         targetL5_3 = torch.tensor([tuple(eval(target)) for target in targetL5_3]).cuda(cudax)

        
        #         print(type(target_F1))
        #         image_F1_list = list(image.to(args.device) for image in image_F1)
        #         targets1 = [{k : v.to(args.device) for k, v in t.items()} for t in target_F1]
        #         image_F2_list = list(image.to(args.device) for image in image_F2)
        #         targets2 = [{k : v.to(args.device) for k, v in t.items()} for t in target_F2]
                
    
         
        #         #########
        #         # outputs = model(f1data,f2data)
        #         losss=model(image_F1, image_F2,image_F1_list, image_F2_list,targets1,targets2)
        #         loss = sum(loss for loss in losss.values())
            

        #         train_loss += loss.item()

        #     print(f'[{epoch + 1:02d}/{args.max_epoch:02d}], train_loss:{train_loss / len(train_loader):.5f}')
        #     losses.append(round(float(train_loss) / len(train_loader), 5))
        #         f1data = sample['f1Data']
        #         f2data = sample['f2Data']
        #         target1 = sample['target1']
        #         target2 = sample['target2']
        #         target3 = sample['target3']
        #         f1data = f1data.cuda(cudax)
        #         f2data = f2data.cuda(cudax)
        #         target1 =torch.tensor([tuple(eval(target)) for target in target1]).cuda(cudax)
        #         target2 = torch.tensor([tuple(eval(target)) for target in target2]).cuda(cudax)
        #         target3 = torch.tensor([tuple(eval(target)) for target in target3]).cuda(cudax)
        #         #########
        #         # outputs = model(f1data, f2data)
        #         outputs1,outputs2,outputs3=model(f1data, f2data)
        #         loss1 = DIYloss(outputs1,target1)
        #         loss2 = DIYloss(outputs2,target2)
        #         loss3 = DIYloss(outputs3,target3)
        #         loss_add = DIYlossadd(outputs1,outputs2,outputs3)

        #         ###########
        #         loss = loss1 + loss2 + loss3 +0.1*loss_add
                
        #         valid_loss += loss.item()

        #         if (step+1)%(len(valiation_dl)) == 0 :
        #             writer_test.add_scalars('test', {
        #                     'test_loss': valid_loss /(step+1),
        #                     'test_loss_val1': loss1.item(),
        #                     'test_loss_val2': loss2.item(),
        #                     'test_loss_val3': loss3.item(),
        #                 }, testIndex)
        #             testIndex += 1

        #     print('epoch:[%d] val_loss: %.3f' % (epoch + 1,  valid_loss /(step+1)))
        #     if epoch >= 10 and valid_loss < max_float:
        #         max_float = valid_loss
        #         print("model--------------saved")
        #         torch.save(model.state_dict(), 'fold-{}_spinmodelresnet0816{}.pth'.format(foldnum,epoch))

        print('Finished Training')
