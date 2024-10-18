from PIL import Image
from flatbuffers.builder import np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class ImageDatasetP2P(Dataset):
    def __init__(self, root, transforms_=None, train=True):
        super().__init__()
        self.transform = transforms.Compose(transforms_)
        self.root = root
        if train:
            path = root + 'catch_pairs.txt'
        # else:
        #     path = root + 'gan_test.txt'

        with open(path, 'r') as f:
            lines = f.readlines()

        self.images_A = []
        self.images_B = []
        # for line in lines[:2400]:
        for line in lines:
            a, b = line.split()

            self.images_A.append(a)
            self.images_B.append(b)

    def __getitem__(self, index):

        image_A = Image.open(self.root + self.images_A[index])
        image_B = Image.open(self.root + self.images_B[index])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        image_A = image_A.resize((256, 256))
        image_B = image_B.resize((256, 256))

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)


        return {"A": item_A, "B": item_B}

    ## 获取A,B数据的长度
    def __len__(self):
        return max(len(self.images_A), len(self.images_B))


class ImageDatasetTestP2P(Dataset):
    def __init__(self, root, transforms_=None, train=True):
        super().__init__()
        self.transform = transforms.Compose(transforms_)
        self.root = root
        if train:
            path = root + 'gan_train.txt'
        else:
            path = root + 'gan_test.txt'
            # path = root + 'false_test.txt'

        with open(path, 'r') as f:
            lines = f.readlines()

        self.images_A = []
        for line in lines:
            a, = line.split()

            self.images_A.append(a)

    def __getitem__(self, index):
        A_name = self.images_A[index]
        image_A = Image.open(self.root + self.images_A[index])

        image_A = image_A.resize((256, 256))
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)

        item_A = self.transform(image_A)

        return {"A": item_A, "name": A_name}

    ## 获取A,B数据的长度
    def __len__(self):
        return len(self.images_A)
