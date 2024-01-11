from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import config


to_pt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((800, 1200))
])


class SuperResDataset(Dataset):
    def __init__(self, datadir=config.DATASET_PATH):
        super().__init__()
        self.data = []
        self.datadir = datadir
        self.image_names = os.listdir(datadir)

        for image_name in self.image_names:
            image_path = os.path.join(self.datadir, image_name)
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.data.append(image_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_high_res = to_pt(image)
        c, h, w = image_high_res.shape
        h, w = h // 4, w // 4
        image_low_res = transforms.Resize((h, w))(image_high_res)
        return image_low_res, image_high_res


def get_dataloader(batch_size=16):
    return DataLoader(SuperResDataset(), batch_size=batch_size)
