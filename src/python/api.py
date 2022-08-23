import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from skimage.transform import resize
from starlette.responses import StreamingResponse

from helpers.imageio_helpers import rescale_image_to_uint8_range
from pipeline import DepthEstimationPipeline

depth_estimation_pipeline = DepthEstimationPipeline()
config = depth_estimation_pipeline.get_configuration()
router = FastAPI()


async def upload_file_to_pipeline_image(upload_file: UploadFile) -> np.ndarray:
    image_raw = await upload_file.read()
    image = np.fromstring(image_raw, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return np.clip(255 * resize(image, config.image_shape) + 0.5, 0, 255).astype(np.uint8)


@router.post("/")
async def run_pipeline(left_view: UploadFile = File(...)):
    left_image = await upload_file_to_pipeline_image(left_view)
    output_disparity = depth_estimation_pipeline.process(left_image, None)
    output_disparity_image = rescale_image_to_uint8_range(output_disparity)
    res, im_png = cv2.imencode(".png", output_disparity_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
