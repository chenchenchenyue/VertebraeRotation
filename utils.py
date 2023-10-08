from PIL import Image
import os
import json
import numpy as np
import cv2
def actionCropImage(idnames,type):
    listsetidnames = []
    for item in idnames:
        data = item+".json"
        listsetidnames.append(data)
    newidnames = list(set(listsetidnames))

    count = 0
    noDatas = []  ####
    '''
    没有标注信息的json文件
    ['caoyue_F1_F1LTorL.json', 'caoyue_F2_F2LTorR.json', 'caoyue_F2_F2UBenL.json', 'caoyue_F2_F2UTorR.json', 'ChengGuangchuan_F2_F2LFlex.json', 'ZhangJianxinyiyou_F1_F1UExten.json', 'ZhangJianxinyiyou_F2_F2UFlex.json']
    '''
    jsonPaths = "output_rename"
    imagePaths= "image_rename_png"
    saveImages = "xxxxxx"

    annotations = []
    images = []

    count_image  = 0
    count_vertbreae = 0
    for itemori in newidnames: #遍历json文件读取X,Y坐标
        for fpath in ["_F1_F1","_F2_F2"]:
            if "2_LiShiqin" in itemori:
                item = "2_LiShiqin"+fpath+itemori.split("_")[2]
            else:
                item = itemori.split("_")[0]+fpath+itemori.split("_")[1]
     
            with open(os.path.join(jsonPaths,item),'r',encoding='utf8') as fp:   #读取json文件
                jsondata=json.load(fp)
                if jsondata["outputs"]!={}:
                    # 读取图像
                    imgData = cv2.imread(os.path.join(imagePaths, item.split('.')[0] + '.jpg'))
                    height, width = imgData.shape[:2]

                    images.append(dict(
                        id=count_image,
                        file_name=item.split(".json")[0]+".jpg",
                        height=height,
                        width=width))
                    bboxes = []
                    labels = []
                    masks = []
                    for polygon in jsondata["outputs"]["object"]:  # 有个别图像画了多个区域
                        name = polygon["name"]
                        data = polygon["polygon"]
                        pts = []
                        pts_x = []
                        pts_y = []
                        for x in range(1,len(data)//2+1):
                            pts_x.append(int(data["x"+str(x)]))
                            pts_y.append(int(data["y"+str(x)]))

                        px = pts_x
                        py = pts_y
                        poly = [(x, y) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]

                        x_min, y_min, x_max, y_max = (
                            min(px), min(py), max(px), max(py))

                        data_anno = dict(
                            image_id=count_image,
                            id=count_vertbreae,
                            category_id=1,
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            area=(x_max - x_min) * (y_max - y_min),
                            segmentation=[poly],
                            iscrowd=0)
                        annotations.append(data_anno)
                        count_vertbreae += 1


                else:
                    noDatas.append(item)

            count_image += 1
    coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{'id':1, 'name': 'S1'}])

    # print(noDatas)

    return coco_format_json



if __name__=="__main__":
    coco_format_json = actionCropImage()

    with open('val_coco_data.json','w') as f:
        json.dump(coco_format_json, f)
    print(json.dumps(coco_format_json))