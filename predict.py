import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = r'C:\Users\111\Desktop\data\gamma\real\rgb\-2.5D\12.jpg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # # read class_indict
    # json_path = r'C:\Users\111\Desktop\pupil\jpg\real\-2.5D\gamma-segment-segment41-11.bmp.png.jpg.jpg-103.jpg'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    #
    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=1).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        predict = torch.squeeze(model(img.to(device))).cpu()
        # predict = torch.nn.liner(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "pred: {}   label: {:.3}".format( predict,
                                                 predict)
    plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(predict,
    #                                               predict))
    plt.show()


if __name__ == '__main__':
    main()
