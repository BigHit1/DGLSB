import PIL.Image
from torch.utils.data import Dataset
from torchvision import transforms
import os


class CelebaHQ512(Dataset):

    # root : 图片所在根目录 datasets/celebaHQ512/image
    # transforms : 图片的预处理 --> Tensor  网络的输入
    def __init__(self, root='./datasets/celebaHQ512/image/', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])) -> None:
        self.root = root
        self.transform = transform
        self.filelist = os.listdir(root)

    def __getitem__(self, item) -> tuple:
        return self.transform(PIL.Image.open(os.path.join(self.root, self.filelist[item]))), 1

    def __len__(self) -> int:
        return len(self.filelist)
