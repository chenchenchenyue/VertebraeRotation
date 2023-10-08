import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import DataLoader
def default_loader(path):

    img=Image.open(path)
    return img
class MedReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""

    def __init__(self, path, data, target, transform=None):
        # 获取训练集文件夹下的所有文件
        self.path = path
        self.data = data
        self.target = target
        self.transform = transform

        # 需要去除未标注数据、缺失数据zhoupingxiangyiyou_F2_F2UTorR
            # ['caoyue_F1_F1LTorL.json', 'caoyue_F2_F2LTorR.json', 'caoyue_F2_F2UBenL.json', 'caoyue_F2_F2UTorR.json', 'ChengGuangchuan_F2_F2LFlex.json', 'ZhangJianxinyiyou_F1_F1UExten.json', 'ZhangJianxinyiyou_F2_F2UFlex.json']
        

        self.numVal = 100.0


        


    def __len__(self):
        return len(self.data)

    # 装载数据，返回[img,label]，idx就是一张一张地读取
    def __getitem__(self, idx):
        nameZhuigu = self.data[idx]   #2_LiShiqin_UExten-L5
        if nameZhuigu.startswith("2"):
            nameFile = "2 LiShiqin"
        else:
            nameFile = nameZhuigu.split("_")[0]  #
        namePart = nameZhuigu.split("-")[0].split("_")[-1]   #UExten
        namePartNum = nameZhuigu.split("-")[1]         #L5

        f1Data = nameFile+"_F1_F1"+namePart+"-"+namePartNum+".jpg"
        f2Data = nameFile+"_F2_F2"+namePart+"-"+namePartNum+".jpg"

        f1Data = default_loader(os.path.join(self.path,f1Data))
        f2Data = default_loader(os.path.join(self.path,f2Data))

        # 处理完毕，将array转换为tensor
  
        f1Data = self.transform(f1Data)
        # data_Sagittal = data_Sagittal.unsqueeze(1)
        f2Data = self.transform(f2Data)
        # data_Coronal = data_Coronal.unsqueeze(3)
        targetVals = tuple(eval(self.target[idx]))







        sample = {'f1Data': f1Data,'f2Data': f2Data, 'target1':str(targetVals[0]), 'target2':str(targetVals[1]), 'target3':str(targetVals[2])}

        return sample


