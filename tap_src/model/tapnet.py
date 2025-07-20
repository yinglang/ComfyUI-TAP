import numpy as np
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
import torch.nn.functional as F
import torch, tqdm


class TAPNetNextModel:
    def __init__(self, ckpt_path='../checkpoints/bootstapnext_ckpt.npz', image_size=(256, 256), device='cuda'):
        model = TAPNext(image_size=image_size)
        model = restore_model_from_jax_checkpoint(model, ckpt_path)
        self.model = model.to(device)
        self.image_size = image_size
        self.device = device
        
    def preprocess(self, video, pts):
        """
            video: list[Image.Image]
            pts: (n, 2), 2 is (x, y)
        Return:
            video: (B=1, T, H, W, 3), H=W=256, normalize to [-1, 1]
            query_points: (B=1, n, 3), 3 is (0, y, x)
            scale_factor: (2,), (scale_w, scale_h)
        """
        img = video[0]
        target_w, target_h = self.image_size
        scale_factor = np.array([target_w / img.width, target_h / img.height])
        pts = (pts * scale_factor)[..., [1, 0]] # (y, x) format
        images = np.array([img.convert('RGB').resize((target_w, target_h)) for img in video])
        images = images / 255 * 2 - 1
        data = {
            "video": images[None, ...], 
            "query_points": np.concatenate([np.zeros((1, len(pts), 1)), pts[None, ...]], axis=-1),
            "scale_factor": scale_factor,
        }
        data = {k: torch.from_numpy(v).to(self.device).float() for k, v in data.items()}  # to cuda
        return data

    def __call__(self, images, points, radius=8, threshold=0.5, use_certainty=False):
        """
        Args:
            video_array: list[Image.Image]
            points: (N, 2) numpy array of query points in the first frame, (x, y) format
            radius: int, radius for certainty calculation
            threshold: float, threshold for visibility logits
            use_certainty: bool, whether to use certainty in visibility prediction
        Return:
            tracks: (T, N, 2) numpy array of tracks, (x, y) format
            occcluded: (T, N) numpy array of occlusion flags
        """
        model = self.model
        # model input and output with format (y, x)
        data = self.preprocess(images, points)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            with torch.no_grad():
                pred_tracks, track_logits, visible_logits, tracking_state = model(
                    video=data['video'][:, :1], query_points=data['query_points']
                )
                pred_visible = visible_logits > 0
                pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
                pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
                    visible_logits.cpu()
                ]
                for frame in tqdm.tqdm(range(1, data['video'].shape[1])):
                    # ***************************************************
                    # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
                    # ***************************************************
                    (
                        curr_tracks,
                        curr_track_logits,
                        curr_visible_logits,
                        tracking_state,
                    ) = model(
                        video=data['video'][:, frame : frame + 1],
                        state=tracking_state,
                    )
                    curr_visible = curr_visible_logits > 0
                    # ***************************************************
                    pred_tracks.append(curr_tracks.cpu())
                    pred_visible.append(curr_visible.cpu())
                    pred_track_logits.append(curr_track_logits.cpu())
                    pred_visible_logits.append(curr_visible_logits.cpu())
                tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
                pred_visible = torch.cat(pred_visible, dim=1).transpose(1, 2)
                track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
                visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)

                pred_certainty = tracker_certainty(tracks, track_logits, radius)
                pred_visible_and_certain = (
                    F.sigmoid(visible_logits) * pred_certainty
                ) > threshold

                if use_certainty:
                    occluded = ~(pred_visible_and_certain.squeeze(-1))
                else:
                    occluded = ~(pred_visible.squeeze(-1))

        tracks = tracks[..., [1, 0]] / data['scale_factor'].to(tracks.device)  # (y, x) to (x, y) format
        return (
            tracks[0].numpy().transpose((1, 0, 2)),  
            occluded[0].numpy().transpose((1, 0)),
        )