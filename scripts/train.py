import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils import train_one_epoch, evaluate
from densenet import DenseNet121

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "dataset")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = 0# = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    num_classess = 5 # 5个类别
    net = DenseNet121(num_classess)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet50-19c8e357.pth"#./weight/resnet18.pth
    

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classess)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    epochs = 15
    best_acc = 0.0
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        mean_loss, train_ac = train_one_epoch(model=net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        #scheduler.step()

        # validate
        val_ac, val_lss = evaluate(model=net,
                       data_loader=val_loader,
                       device=device)
        train_lss = mean_loss

        print("[epoch {}] 训练精度train_accuracy: {}".format(epoch, round(train_ac, 3)))
        print("[epoch {}] 训练损失train_loss: {}".format(epoch, round(train_lss, 3)))
        print("[epoch {}] 验证精度val_accuracy: {}".format(epoch, round(val_ac, 3)))
        print("[epoch {}] 验证损失val_loss: {}".format(epoch, round(val_lss, 3)))
        train_loss.append(train_lss)
        train_acc.append(train_ac)
        val_acc.append(val_ac)
        val_loss.append(val_lss)
        if val_ac > best_acc:
            best_acc = val_ac
            torch.save(net.state_dict(),"./output/best.pth")
        torch.save(net.state_dict(), "./output/model-{}.pth".format(epoch))

    with open("./output/train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))
    with open("./output/val_loss.txt", 'w') as val_los:
        val_los.write(str(val_loss))
    with open("./output/train_acc.txt", 'w') as train:
        train.write(str(train_acc))
    with open("./output/val_acc.txt", 'w') as val:
        val.write(str(val_acc))

    print('Finished Training, best acc is '+str(best_acc))


if __name__ == '__main__':
    main()
