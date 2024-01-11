import os
import torch
import model as arch


def get_model():
    state_dict = torch.load(
        './pretrained_model/RRDB_ESRGAN_x4.pth')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(state_dict)

    return model
