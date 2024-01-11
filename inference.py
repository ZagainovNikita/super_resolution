import torch
from load_pretrained import get_model
from typing import Literal
from model import RRDBNet
import os
import cv2
import numpy as np
import config
from tqdm import tqdm


def init_model_with_weights(weights: Literal["original", "checkpoint"]):
    if weights == "original":
        return get_model()

    if weights == "checkpoint":
        checkpoint = max(os.listdir("train_checkpoints"))
        model_path = os.path.join("train_checkpoints", checkpoint)
        model = RRDBNet()
        model.load_state_dict(torch.load(model_path))
        return model

    raise ValueError("incorrect weights")


def upscale(model, images_path="test_images", save_path="results"):
    for image in tqdm(os.listdir(images_path)):
        image_path = os.path.join(images_path, image)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(
            img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(config.DEVICE)

        with torch.no_grad():
            output = model(img_LR).data.squeeze(
            ).float().cpu().clamp(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        cv2.imwrite(os.path.join(save_path, image), output)


def main():
    model = init_model_with_weights("original")
    upscale(model)


if __name__ == "__main__":
    main()
