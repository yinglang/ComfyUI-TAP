# ComfyUI-TAP

```sh
git clone https://github.com/deepmind/tapnet.git
cd tapnet && pip install .
pip install flax jax jaxlib

mkdir -p checkpoints
wget -P checkpoints --no-check-certificate https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/tapnext/tapnext_ckpt.npz
```

# Usage Example

```py
from model.tapnet import TAPNetNextModel
import utils.vis as vis
import numpy as np

tap_model = TAPNetNextModel('../checkpoints/bootstapnext_ckpt.npz', image_size=(256, 256), device='cuda:0')

webp_path = "./00000003_000_001_i2v_00001_.webp"
images = video_utils.WebpReader(webp_path).images()
pts = np.array([[210, 724], [214, 477]])

tracks, _ = tap_model(images, pts)
vis.add_points_to_frames(images, tracks, radius=6)
# images[0].save("")

images_array = np.array([img.convert('RGB') for img in images])
vis.vis_omnimotion_style(images_array, tracks, tracks[:, :1], '/tmp/result.webp')
```

# Thanks

- [TAPNet serial](https://github.com/google-deepmind/tapnet)
- [co-tracker](https://github.com/facebookresearch/co-tracker)
- [Tracking Everything Everywhere All at Once](https://omnimotion.github.io/)