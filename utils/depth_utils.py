import torch
from utils.dtof_simulator import DToFSimulator
import numpy as np

tof_simulator = DToFSimulator(scale=16, temp_res=1024, dtof_sampler='peak', with_conf=False, num_peaks = 1, threshold = 0.1, key='lq')

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    # tof simulator
    # prediction[prediction<0] = 0
    # d = prediction.cpu().numpy()
    # d_max = d.max()
    # tof_depth = tof_simulator.tof_mean(d / d_max, img.permute(1,2,0).cpu().numpy())
    # return torch.tensor(tof_depth[...,0]).cuda()
    
    d_max = prediction.max()
    tof_depth = tof_simulator.tof_mean(prediction / d_max, img.permute(1,2,0))
    return tof_depth[...,0]

    # return prediction

