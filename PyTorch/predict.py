import os

import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

import transforms as T


def change_input_channels(model, input_channels):
    print(model.backbone.conv1)
    model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    print(model.backbone.conv1)
    return model

def predict(model, device, data_path):
    model.eval()
    mean=(0.485, 0.456, 0.406, 0.405)
    std=(0.229, 0.224, 0.225, 0.217)
    
    image_path = os.path.join(data_path, 'images')
    output_path = os.path.join(data_path, 'results')
    image_list = [os.path.join(image_path,i) for i in os.listdir(image_path)]

    for image_path in tqdm(image_list):
        image = Image.open(image_path)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=mean, std=std)
        image = image.unsqueeze(0)
        image = image.to(device)

        score = model(image)['out'][0]
        output_predictions = score.argmax(0)
        mask = output_predictions.byte().cpu().numpy()
        mask += 1
        mask = Image.fromarray(mask)

        output = os.path.basename(image_path).split('.')[0] + ".png"
        output = os.path.join(output_path, output)
        # cv2.imwrite(output, mask)
        mask.save(output)

def main(args):
    num_classes = 4
    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained)
    # change_input_channels(model, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    predict(model, device, args.data_path)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Predicting')

    parser.add_argument('--data-path', default='/data', help='dataset path')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=8, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./model', help='path where to save')
    parser.add_argument('--resume', default='model\model_30.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
