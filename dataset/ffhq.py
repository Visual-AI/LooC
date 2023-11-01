# we have appointed the first 60,000 images to be used for training and the remaining 10,000 for validation


import pathlib
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class ImagesFolder(Dataset):
    def __init__(self, root, transform=None,distributed=False,open_mode="RGB", split='train'):
        self.images_path = self.getImagesPath(root,distributed)

        if split == 'train':
            self.images_path = self.images_path[0:60000]  # 60k
        elif split == 'val':
            self.images_path = self.images_path[0:-10000]  # 10k

        self.transform=transform
        self.open_mode = open_mode

    def getImagesPath(self,root,distributed=False):
        if distributed:
            images_path=list(pathlib.Path(root).rglob('*.png'))
        else:
            images_path = list(pathlib.Path(root).glob('*.png'))
        return images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        path = self.images_path[index]
        image = Image.open(path)
        if self.open_mode is not None: image = image.convert(self.open_mode)
        if self.transform is not None: image = self.transform(image)
        return image, 0


if __name__ == '__main__':
    # 调整图片大小，转化为张量，调整值域为-1到1
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据集
    train_ds=ImagesFolder(r'E:\Data\FFHQ\images1024x1024',transform)
    # 观察数据集
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np

    train_dl = DataLoader(train_ds, batch_size=16)
    for images, labels in train_dl:
        images = make_grid(images, nrow=4)
        images = np.transpose(images.data * 0.5 + 0.5, [1, 2, 0])
        plt.imshow(images)
        plt.show()
