# 交叉验证    
############################################
from itertools import count
from sklearn.model_selection import KFold
from train_crossValidation_content import mainTrain
from utils import actionCropImage
import json
path = "./new裁剪图片/"
dataSetdata = []
dataSetlabel = []
noSaveDatas = ['caoyue_LTorL', 'caoyue_LTorR', 'caoyue_UBenL', 'caoyue_UTorR', 'ChengGuangchuan_LFlex', 'ZhangJianxinyiyou_UExten', 'ZhangJianxinyiyou_UFlex','zhoupingxiangyiyou_UTorR']
with open("angleoutput.txt","r",encoding="utf8") as fdataset:
    datas = fdataset.readlines()
    for data in datas:
        dataLine = data.strip().split("\t")
        if dataLine[0].split("-")[0] not in noSaveDatas:
            dataSetdata.append(dataLine[0].split("-")[0])


dataSetdata = list(set(dataSetdata))
print(len(dataSetdata))
FOLD = 5
sfolder = KFold(n_splits=FOLD,shuffle=True,random_state=24)
tr_folds = []
val_folds = []
countFold = 1
for train_idx, val_idx in sfolder.split(dataSetdata):
    tr_folds.append(train_idx)
    val_folds.append(val_idx)
    print("--------------------train fold {}--------------------".format(countFold))
    with open('cocodata/train_{}.json'.format(countFold),'w') as f:
        json.dump(actionCropImage([dataSetdata[i] for i in train_idx],"train"), f)
    
    with open('cocodata/val_{}.json'.format(countFold),'w') as f:
        json.dump(actionCropImage([dataSetdata[i] for i in val_idx],"test"), f)
    mainTrain(countFold)
    countFold += 1

