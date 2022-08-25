import torch
import cuda_imageops
import cuda_depth
import torchvision.transforms
from PIL import Image

T = torchvision.transforms.PILToTensor()
P = torchvision.transforms.ToPILImage()


def main():
    image = Image.open("../data/left.png").convert("RGB")
    torch_image = T(image).cuda().float()
    # output = cuda_imageops.rgb_to_grayscale(torch_image.contiguous())
    # output = cuda_imageops.mean_pool(output, 2)
    # print(output.shape)
    # P(output.byte()).show()
    stereo_matching = cuda_depth.StereoMatching()
    output = stereo_matching.compute_disparity_map(torch_image.contiguous(), torch_image.contiguous())
    print(output.device)


if __name__ == "__main__":
    main()
