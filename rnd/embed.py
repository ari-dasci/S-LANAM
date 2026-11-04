"""Frozen pre-trained encoders of H3. All embeddings are L2-normalised."""
import os
import numpy as np
import torch

from . import config as C


def _l2(E):
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


class ImageEncoder:                       # CLIP ViT-B/32 (open_clip, OpenAI weights), 512-d
    def __init__(self, device):
        import open_clip
        cfg = C.ENCODERS["IMG"]
        # OpenAI weights were trained with QuickGELU; open_clip 3.x builds GELU unless forced
        self.model, _, self.pre = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"],
                                                                        force_quick_gelu=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.revision = f"open_clip:{cfg['arch']}:{cfg['pretrained']}:{open_clip.__version__}"

    @torch.no_grad()
    def __call__(self, X_uint8, bs=512):
        from PIL import Image
        out = []
        for b in range(0, len(X_uint8), bs):
            x = torch.stack([self.pre(Image.fromarray(im)) for im in X_uint8[b:b + bs]]).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                out.append(self.model.encode_image(x).float().cpu().numpy())
        return _l2(np.concatenate(out))


class TextEncoder:                        # E5-large-v2 with "query: " prefix, mean pooling, 1024-d
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        cfg = C.ENCODERS["TXT"]
        self.model = SentenceTransformer(cfg["name"], device=device)
        self.model.max_seq_length = cfg["max_len"]
        self.prefix = cfg["prefix"]
        self.revision = f"{cfg['name']}"

    def __call__(self, texts, bs=128):
        E = self.model.encode([self.prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return _l2(E)


class AudioEncoder:                       # PANNs CNN14 (AudioSet checkpoint), 32 kHz, 2048-d
    def __init__(self, device):
        from panns_inference import AudioTagging
        ckpt = os.path.join(os.path.expanduser("~"), "panns_data", "Cnn14_mAP=0.431.pth")
        self.at = AudioTagging(checkpoint_path=ckpt, device="cuda" if device.startswith("cuda") else "cpu")
        self.min_len = C.ENCODERS["AUD"]["min_len"]
        self.revision = "panns_inference:Cnn14_mAP=0.431"

    def __call__(self, waves32):
        out = []
        for w in waves32:                 # one clip at a time: no padding beyond the minimum length of 1 s
            w = np.asarray(w, dtype=np.float32)
            if len(w) < self.min_len:
                w = np.pad(w, (0, self.min_len - len(w)))
            _, emb = self.at.inference(w[None, :])
            out.append(emb[0])
        return _l2(np.stack(out))


ENCODER_CLASSES = {"IMG": ImageEncoder, "TXT": TextEncoder, "AUD": AudioEncoder}


def get_encoder(ds, device):
    return ENCODER_CLASSES[ds](device)
