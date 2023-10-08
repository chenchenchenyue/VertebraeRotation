import torch
import numpy as np
import cv2
import argparse
from PIL import Image

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import os
transform = transforms.Compose(
    [
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.37149432, 0.37149432,0.37149432),(0.27260366, 0.27260366, 0.27260366))
    ])

def getAngel():
    dataSetdata = []
    dataSetlabel = []
    noSaveDatas = ['caoyue_LTorL', 'caoyue_LTorR', 'caoyue_UBenL', 'caoyue_UTorR', 'ChengGuangchuan_LFlex', 'ZhangJianxinyiyou_UExten', 'ZhangJianxinyiyou_UFlex','zhoupingxiangyiyou_UTorR']
    with open("angleoutput.txt","r",encoding="utf8") as fdataset:
        datas = fdataset.readlines()
        for data in datas:
            dataLine = data.strip().split("\t")
            if dataLine[0].split("-")[0] not in noSaveDatas:
                dataSetdata.append(dataLine[0])
                dataSetlabel.append(dataLine[1])
    return dataSetdata,dataSetlabel
class MedicalDataset(Dataset) :
    def __init__(self, foldnum,datssetmode,args):
        super(MedicalDataset, self).__init__()
        # args 是传入的所有参数集合
        self.args = args
        # 加载图片和图片的注释数据，也即分割对象的 masks、labels、boxes
        ann_file = os.path.join(args.data_dir, "{}_{}.json".format(datssetmode,foldnum))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        self.transform = transform
        self.angeldatas,self.angellabels = getAngel()
    def coco(self):
       
        return self.coco
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.args.data_dir, img_info["file_name"]))
        ##################
        imgs = cv2.imread(os.path.join(self.args.data_dir, img_info["file_name"]))
        imgs = cv2.resize(imgs, (1024,1024))
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        ##################
        return self.transform(image.convert("RGB")),imgs
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    def __getitem__(self, i):
        
        img_id = self.ids[i]
        image,ori_image = self.get_image(img_id)

        target = self.get_target(img_id)


        ##############
        img_metadata = self.coco.dataset['images']
        name = self.coco.imgs[int(img_id)]["file_name"]

        
        # caoyue_F1_F1UBenR.jpg
        if "_F1_F1" in name:
            angelNameL3 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F1_F1")[0]+"_"+name.split(".jpg")[0].split("_F1_F1")[1]+"-L3")]))
            angelNameL4 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F1_F1")[0]+"_"+name.split(".jpg")[0].split("_F1_F1")[1]+"-L4")]))
            angelNameL5 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F1_F1")[0]+"_"+name.split(".jpg")[0].split("_F1_F1")[1]+"-L5")]))

            
            newfile_name = name.split("_F1_F1")[0]+"_F2_F2"+name.split("_F1_F1")[1]
            img_metadata = [img for img in img_metadata if img['file_name'] == newfile_name]
            # if len(img_metadata)<1:
            #     print(name,newfile_name)
            #############
            image_F2,ori_image_F2 = self.get_image(img_metadata[0]["id"])
            target_F2 = self.get_target(img_metadata[0]["id"])

            img_id_F1 = img_id
            img_id_F2 = img_metadata[0]["id"]
            
 
                        
      
            return image,target,ori_image,image_F2,target_F2 , ori_image_F2,str(angelNameL3[0]),str(angelNameL3[1]),str(angelNameL3[2]),str(angelNameL4[0]),str(angelNameL4[1]),str(angelNameL4[2]),str(angelNameL5[0]),str(angelNameL5[1]),str(angelNameL5[2]),name,img_id_F1,img_id_F2

        if "_F2_F2" in name:
            angelNameL3 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F2_F2")[0]+"_"+name.split(".jpg")[0].split("_F2_F2")[1]+"-L3")]))
            angelNameL4 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F2_F2")[0]+"_"+name.split(".jpg")[0].split("_F2_F2")[1]+"-L4")]))
            angelNameL5 = tuple(eval(self.angellabels[self.angeldatas.index(name.split(".jpg")[0].split("_F2_F2")[0]+"_"+name.split(".jpg")[0].split("_F2_F2")[1]+"-L5")]))

            newfile_name = name.split("_F2_F2")[0]+"_F1_F1"+name.split("_F2_F2")[1]
            img_metadata = [img for img in img_metadata if img['file_name'] == newfile_name]
            # if len(img_metadata)<1:
            #     print(name,newfile_name)
            #############
            image_F1,ori_image_F1 = self.get_image(img_metadata[0]["id"])
            target_F1 = self.get_target(img_metadata[0]["id"])


            img_id_F1 = img_metadata[0]["id"]
            img_id_F2 = img_id

                        
            return image_F1, target_F1 , ori_image_F1,image,target , ori_image,str(angelNameL3[0]),str(angelNameL3[1]), str(angelNameL3[2]),str(angelNameL4[0]), str(angelNameL4[1]), str(angelNameL4[2]),str(angelNameL5[0]), str(angelNameL5[1]), str(angelNameL5[2]),name,img_id_F1,img_id_F2

        

    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=tuple, default=(600, 600))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_dir', type=str, default='LabPicsMedical/Train')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:8" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_epoch', type=int, default=200)

    args = parser.parse_args()
    data = MedicalDataset(args)
