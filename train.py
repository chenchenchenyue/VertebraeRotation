import torch
import torch.optim as optim
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import MedicalDataset

class Solver :
    def __init__(self, args):
        self.args = args
        # 制作训练集，要注意输出的数据状态，分割任务中因为每张图片的目标数量可能不一样，因此无法进行张量拼接
        # 因此，对于图片或者标注信息我们要获取列表型数据，由参数 collate_fn 决定，可以查查该参数的用法
        self.train_loader = DataLoader(MedicalDataset(args), batch_size = args.batch_size, shuffle = True,
                                       collate_fn = lambda x: tuple(zip(*x)))

    def train(self):
        # 具有 ResNet-50-FPN 主干的 maskrcnn 的预训练模型
        model = maskrcnn_resnet50_fpn(pretrained = True)
        # 更换分类器
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes = self.args.num_classes)
        model = model.to(self.args.device)
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
        model.train()

        losses = []
        min_loss = 100000
        # 训练
        for epoch in range(self.args.max_epoch) :
            train_loss = 0.0
            for images, targets,ori_image in self.train_loader :
                images = list(image.to(self.args.device) for image in images)
                targets = [{k : v.to(self.args.device) for k, v in t.items()} for t in targets]
                
                # 损失，如果输入了 target 则输出损失，否则输出的是预测分数、框、分割等等信息
                output = model(images, targets)
                loss = sum(loss for loss in output.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f'[{epoch + 1:02d}/{self.args.max_epoch:02d}], train_loss:{train_loss / len(self.train_loader):.5f}')
            losses.append(round(float(train_loss) / len(self.train_loader), 5))
            
            # 保存 fine-tuning 后的模型
            model_save_path = Path(self.args.model_save_path)
            if not model_save_path.exists() :
                model_save_path.mkdir(parents = True, exist_ok = True)

            if train_loss < min_loss :
                min_loss = train_loss
                torch.save(model.state_dict(), model_save_path.joinpath(f'mask_rcnn_with_{self.args.max_epoch}epochs.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type = tuple, default = (1024, 1024))
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--data_dir', type = str, default = 'cocodata')
    parser.add_argument('--data_set', type = str, default = 'train')

    parser.add_argument('--num_classes', type = int, default = 2)
    parser.add_argument('--device', type = str, default = "cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--model_save_path', type = str, default = 'checkpoint')

    args = parser.parse_args()
    solver = Solver(args)
    solver.train()
