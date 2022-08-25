import io

import cv2
import torch
import torchvision.io
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

from helpers.imageio_helpers import rescale_image_to_uint8_range
from pipeline import DepthEstimationPipeline

depth_estimation_pipeline = DepthEstimationPipeline()
config = depth_estimation_pipeline.get_configuration()
router = FastAPI()


async def upload_file_to_pipeline_image(upload_file: UploadFile) -> torch.Tensor:
    image_buffer = await upload_file.read()
    buffer = torch.frombuffer(bytearray(image_buffer), dtype=torch.uint8)
    resize_transform = T.Resize(size=config.image_shape)
    image = torchvision.io.decode_png(buffer)
    return resize_transform(image)


@router.post("/")
async def run_pipeline(left_view: UploadFile = File(...)):
    left_image = await upload_file_to_pipeline_image(left_view)
    output_disparity = depth_estimation_pipeline.process(left_image, None)
    output_disparity_image = rescale_image_to_uint8_range(output_disparity.cpu().numpy())
    _, im_png = cv2.imencode(".png", output_disparity_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(router, host="localhost", port=8080)
