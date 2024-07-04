
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.models import mobilenet_v2
# Load model directly

import qai_hub as hub
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model.eval()
print(type(model))

# Using pre-trained MobileNet
# torch_model = mobilenet_v2(pretrained=True)
# torch_model.eval()

# # Step 1: Trace model
# input_shape = (1, 3, 224, 224)
# example_input = torch.rand(input_shape)
# traced_torch_model = torch.jit.trace(torch_model, example_input)

# # Step 2: Compile model
# device = hub.Device("Samsung Galaxy S24 (Family)")
# compile_job = hub.submit_compile_job(
#     model=traced_torch_model,
#     device=device,
#     input_specs=dict(image=input_shape),
# )
# assert isinstance(compile_job, hub.CompileJob)

# # Step 3: Profile on cloud-hosted device
# target_model = compile_job.get_target_model()
# assert isinstance(target_model, hub.Model)
# profile_job = hub.submit_profile_job(
#     model=target_model,
#     device=device,
# )

# # Step 4: Run inference on cloud-hosted device
# sample_image_url = (
#     "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
# )
# response = requests.get(sample_image_url, stream=True)
# response.raw.decode_content = True
# image = Image.open(response.raw).resize((224, 224))
# input_array = np.expand_dims(
#     np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
# )

# # Run inference using the on-device model on the input image
# inference_job = hub.submit_inference_job(
#     model=target_model,
#     device=device,
#     inputs=dict(image=[input_array]),
# )
# assert isinstance(inference_job, hub.InferenceJob)

# # Download inference output dict[str, [np.array]]
# # where str: name of the output
# # [np.array]: output as a batch of numpy.array
# on_device_output = inference_job.download_output_data()
# assert isinstance(on_device_output, dict)

# # Step 5: Post-processing the on-device output
# output_name = list(on_device_output.keys())[0]
# out = on_device_output[output_name][0]
# on_device_probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

# # Read the class labels for imagenet
# sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
# response = requests.get(sample_classes, stream=True)
# response.raw.decode_content = True
# categories = [str(s.strip()) for s in response.raw]

# # Print top five predictions for the on-device model
# print("Top-5 On-Device predictions:")
# top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
# for c in reversed(top5_classes):
#     print(f"{c} {categories[c]:20s} {on_device_probabilities[0][c]:>6.1%}")

# # Step 6: Download model
# model = compile_job.download_target_model()