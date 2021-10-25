# %%
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
import glob
import os
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

basic_transforms = {
    "train": nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ),
    "test": nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ),
}


# %%
class MyDataSet:
    def __init__(self, file_list, shuffle=False, random_seed=0):
        self.file_list = file_list
        self.shuffle = shuffle
        self.data_length = len(self.file_list)

        np.random.seed(random_seed)
        if shuffle:
            self.ref_index = np.random.permutation(self.data_length)
        else:
            self.ref_index = np.arange(self.data_length)

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        img = read_image(self.file_list[self.ref_index[index]]["img_path"])
        label = self.file_list[self.ref_index[index]]["label"]
        return img / 1.0, label


class ModelIIC(nn.Module):
    def __init__(self):
        super(ModelIIC, self).__init__()
        self.model_conv = models.mobilenet_v2(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False

        num_ftrs = self.model_conv.classifier[1].in_features
        self.model_conv.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, 10, bias=True),
        )

    def forward(self, x):
        x = F.softmax(self.model_conv(x), dim=1)
        return x


def create_dataset(dataset_dir):
    dataset = {}

    _file_list = glob.glob(
        os.path.join(dataset_dir, "unlabeled", "*", "*.png"),
        recursive=True,
    )
    train_file_list = [{"img_path": p, "label": 0} for i, p in enumerate(_file_list)]
    dataset["train"] = MyDataSet(train_file_list, shuffle=False)

    test_file_list = []
    for label in label_map.keys():
        _file_list = glob.glob(os.path.join(dataset_dir, "test", str(label), "*.png"))
        _data = [{"img_path": p, "label": label} for i, p in enumerate(_file_list)]
        test_file_list.extend(_data)
    dataset["test"] = MyDataSet(test_file_list, shuffle=False)

    return dataset


def random_transform(x):
    """
    1st step: scaling, triming, rotation
    2nd step: color transformation, horizontal flip
    3rd step: add noise(option)
    """
    trans1 = torchvision.transforms.RandomAffine(10, (0.2, 0.4), (0.6, 0.75))
    trans2 = torchvision.transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
    )
    trans3 = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    _x = x.clone()
    _x = trans1(_x)
    _x = trans2(_x)
    _x = trans3(_x)
    return _x


def IID_loss(phi_x, phi_x_prime, EPS=sys.float_info.epsilon):
    _, k = phi_x.shape
    p_cc = compute_joint(phi_x, phi_x_prime)  # k x k matrix
    p_c = p_cc.sum(dim=1).view(k, 1).expand(k, k)
    p_c_prime = p_cc.sum(dim=0).view(1, k).expand(k, k)
    loss = -p_cc * (torch.log(p_cc) - torch.log(p_c) - torch.log(p_c_prime))

    p_cc = torch.where(p_cc < EPS, torch.tensor([EPS], device=p_cc.device), p_cc)
    p_c = torch.where(p_c < EPS, torch.tensor([EPS], device=p_c.device), p_c)
    p_c_prime = torch.where(
        p_c_prime < EPS, torch.tensor([EPS], device=p_c_prime.device), p_c_prime
    )

    loss = loss.sum()
    return loss


def compute_joint(phi_x, phi_x_prime):
    p_cc = phi_x.unsqueeze(2) * phi_x_prime.unsqueeze(1)
    p_cc = p_cc.sum(dim=0)
    p_cc = (p_cc + p_cc.t()) / 2  # symmetrise
    p_cc = p_cc / p_cc.sum()  # normalize
    return p_cc


def train(epoch, model, train_loader, optimizer, device, base_transform):
    model.train()
    n_batch = len(dataset["train"]) // batch_size
    for ep in range(1, epoch + 1):
        loss_ep = []
        for i, (x, _) in enumerate(train_loader, start=1):
            x = base_transform(x)
            x_prime = random_transform(x)

            x = x.to(device)
            x_prime = x_prime.to(device)

            optimizer.zero_grad()

            phi_x = model(x)
            phi_x_prime = model(x_prime)

            loss = IID_loss(phi_x, phi_x_prime)
            loss_ep.append(loss.detach())

            loss.backward()
            optimizer.step()
            print(
                "\repoch=%03d/%03d, batch=%03d/%03d, loss=%.5f, loss_ep=%.5f"
                % (ep, epoch, i, n_batch, loss, np.mean(loss_ep)),
                end="",
            )
        print("")
        print("epoch=%03d/%03d, loss=%.5f" % (ep, epoch, np.mean(loss_ep)))

    return model, optimizer


def test(model, test_loader, device, base_transform):
    model.eval()
    with torch.no_grad():
        pred = []
        true = []
        n_batch = len(dataset["test"]) // batch_size + 1
        for i, (x, label) in enumerate(test_loader, start=1):
            x = base_transform(x)
            x = x.to(device)

            optimizer.zero_grad()
            phi_x = model(x)
            _pred = torch.argmax(phi_x, dim=1)

            pred.extend(_pred.numpy())
            true.extend(label.numpy())
            print("\rcalculating...batch=%03d/%03d" % (i, n_batch), end="")
    print("...Done!")

    return np.array(pred), np.array(true)


# %%
label_map = {
    0: "unlabeled",
    1: "airplane",
    2: "bird",
    3: "car",
    4: "cat",
    5: "deer",
    6: "dog",
    7: "horse",
    8: "monkey",
    9: "ship",
    10: "truck",
}
dataset_dir = "/Users/yohei/workspace/dataset/STL-10/"
dataset = create_dataset(dataset_dir)
# %%
# train data
img, label = dataset["train"].__getitem__(0)
plt.imshow(img.permute(1, 2, 0))
plt.title(label_map[label])

# %%
# test data
img, label = dataset["test"].__getitem__(0)
plt.imshow(img.permute(1, 2, 0))
plt.title(label_map[label])


# %%
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    dataset=dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=dataset["test"],
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

# print("Train")
# for i, (img, _) in enumerate(train_loader):
#     print(img.shape)
#     if i > 3:
#         break

# print("Test")
# for i, (img, label) in enumerate(test_loader):
#     print(img.shape)
#     print(label.shape)
#     img_prime = random_transform(img)

#     fig, ax = plt.subplots(ncols=2, nrows=1)
#     ax[0].imshow(img[0].permute(1, 2, 0))
#     ax[1].imshow(img_prime[0].permute(1, 2, 0))
#     plt.title(label_map[int(label[0])])
#     if i > 3:
#         break

# %%
device = "cpu"
epoch = 3
model = ModelIIC()
model.to(device)
optimizer = torch.optim.Adam(model.model_conv.classifier.parameters(), lr=1e-4)
# %%
train_set = {
    "epoch": epoch,
    "model": model,
    "train_loader": train_loader,
    "optimizer": optimizer,
    "device": device,
    "base_transform": basic_transforms["train"],
}
model, _ = train(**train_set)

# %%
test_set = {
    "model": model,
    "test_loader": test_loader,
    "device": device,
    "base_transform": basic_transforms["test"],
}
pred, true = test(**test_set)
true = true - 1

# %%
matrix = np.zeros((10, 10))

for i in range(len(pred)):
    row = true[i]
    col = pred[i]
    matrix[row][col] += 1

np.set_printoptions(suppress=True)
print(matrix)

# %%
torch.save(model.state_dict(), "./model/resnet/model.pt")

# %%
