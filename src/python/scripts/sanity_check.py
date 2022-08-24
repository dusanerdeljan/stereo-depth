import torch
import cuda_imageops
import torchvision.transforms
from PIL import Image

T = torchvision.transforms.PILToTensor()
P = torchvision.transforms.ToPILImage()


def main():
    image = Image.open("../data/left.png").convert("RGB")
    torch_image = T(image).cuda().float()
    output = cuda_imageops.rgb_to_grayscale(torch_image.contiguous())
    P(output.byte()).show()


if __name__ == "__main__":
    main()
