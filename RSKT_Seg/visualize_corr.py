import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os


def visualize_corr(corr_1, files_name, save_prefix='./output_vis_cost/'):
    file_name = os.path.basename(files_name)
    file_base_name = os.path.splitext(file_name)[0]

    cv2_img = Image.open(files_name)
    cv2_img = cv2.cvtColor(np.array(cv2_img), cv2.COLOR_RGB2BGR)

    for b in range(corr_1.shape[1]):
        H, W = cv2_img.shape[:2]

        vis_1 = corr_1.squeeze(1)[:, b, :, :]
        vis_1 = (vis_1 - vis_1.min()) / (vis_1.max() - vis_1.min())
        vis = F.interpolate(vis_1.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)
        vis = vis.permute(1, 2, 0).cpu().detach().numpy()

        vis = (vis * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        vis = cv2_img * 0.4 + vis * 0.6

        save_dir = os.path.join(save_prefix, file_base_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, '{}_{}.png'.format(file_name,b))
        cv2.imwrite(save_path, vis)    
