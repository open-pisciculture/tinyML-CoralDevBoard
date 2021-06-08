# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

```
bash examples/install_requirements.sh semantic_segmentation.py

python3 examples/semantic_segmentation.py \
  --model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
  --keep_aspect_ratio \
  --output ${HOME}/segmentation_result.jpg
```
"""

import argparse

import numpy as np
from PIL import Image
import cv2
import time

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True,
                      help='Path of the segmentation model.')
  parser.add_argument('--input', required=False,
                      help='File path of the input image.')
  parser.add_argument('--output', default="${HOME}/semantic_segmentation_result.jpg",
                      help='File path of the output image.')
  parser.add_argument(
      '--keep_aspect_ratio',
      action='store_true',
      default=False,
      help=(
          'keep the image aspect ratio when down-sampling the image by adding '
          'black pixel padding (zeros) on bottom or right. '
          'By default the image is resized and reshaped without cropping. This '
          'option should be the same as what is applied on input images during '
          'model training. Otherwise the accuracy may be affected and the '
          'bounding box of detection result may be stretched.'))
  args = parser.parse_args()

  interpreter = make_interpreter(args.model, device=':0')
  interpreter.allocate_tensors()
  width, height = common.input_size(interpreter)

  ################ Reading from USB Cam ################
  cap = cv2.VideoCapture(1) # For the moment we use index 1 for USB Cam
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      cv2_im = frame # <--- This is the image

      cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)


      t0 = time.time()
      # #  Start inference process ---> # #

      # img = Image.open(args.input) # Opening image from path
      img = Image.fromarray(cv2_im_rgb) # Converting to PIL Image
      if args.keep_aspect_ratio:
        resized_img, _ = common.set_resized_input(
            interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
      else:
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        common.set_input(interpreter, resized_img)

      interpreter.invoke()

      result = segment.get_output(interpreter)
      if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

      # If keep_aspect_ratio, we need to remove the padding area.
      new_width, new_height = resized_img.size
      result = result[:new_height, :new_width]
      mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

      # Concat resized input image and processed segmentation results.
      output_img = Image.new('RGB', (2 * new_width, new_height))
      output_img.paste(resized_img, (0, 0))
      output_img.paste(mask_img, (width, 0))
      # print('Done. Results saved at', args.output) # For the moment we won't save the results

      # # <--- End inference process # #
      t1 = time.time()

      cv2_im = cv2.cvtColor( np.array(output_img), cv2.COLOR_RGB2BGR ) # Converting to cv2 Image

      cv2.putText(cv2_im, 'FPS:'+str(int(1/(t1-t0))), (100, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
      cv2.imshow('frame', cv2_im)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  output_img.save(args.output)
  cap.release()
  cv2.destroyAllWindows()



if __name__ == '__main__':
  main()
