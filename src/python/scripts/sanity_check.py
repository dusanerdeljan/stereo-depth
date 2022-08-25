import numpy as np
import torch
import cuda_imageops
import cuda_depth
import torchvision.transforms
from PIL import Image

T = torchvision.transforms.PILToTensor()
P = torchvision.transforms.ToPILImage()


def main():
    left_image = T(Image.open("../data/left.png").convert("RGB")).cuda().float().contiguous()
    right_image = T(Image.open("../data/right.png").convert("RGB")).cuda().float().contiguous()
    # output = cuda_imageops.rgb_to_grayscale(torch_image.contiguous())
    # output = cuda_imageops.mean_pool(output, 2)
    # print(output.shape)
    # P(output.byte()).show()
    stereo_matching = cuda_depth.StereoMatching()
    output = stereo_matching.compute_disparity_map(left_image, right_image)
    print(output.shape)
    Image.fromarray((output * 256).cpu().numpy().astype(np.uint16)).show()


if __name__ == "__main__":
    main()
