"""Zero-shot, deterministic annotators of H4.

Each annotator labels the clean training set once; labels, scores of every class and the model
revision are stored in RUNS/noise/<DS>_F.npz together with r_F and T_F.
"""
import os
import numpy as np
import torch

from . import config as C
from . import noise as N


def _hf_revision(repo):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception as e:                         # offline: record the local snapshot name instead
        return f"unknown ({type(e).__name__})"


@torch.no_grad()
def annotate_img(X_uint8, device, bs=512, cfg=None):
    import open_clip
    from PIL import Image
    cfg = cfg or C.ANNOTATORS["IMG"]
    model, _, pre = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"],
                                                          force_quick_gelu=True)   # as the OpenAI weights
    tok = open_clip.get_tokenizer(cfg["arch"])
    model = model.to(device).eval()
    t = model.encode_text(tok([cfg["template"].format(c) for c in C.CIFAR10_CLASSES]).to(device)).float()
    t = t / t.norm(dim=-1, keepdim=True)
    scores = []
    for b in range(0, len(X_uint8), bs):
        x = torch.stack([pre(Image.fromarray(im)) for im in X_uint8[b:b + bs]]).to(device)
        f = model.encode_image(x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        scores.append((f @ t.T).cpu().numpy())
    S = np.concatenate(scores)
    return S.argmax(1), S, f"open_clip:{cfg['arch']}:{cfg['pretrained']}:{open_clip.__version__}"


@torch.no_grad()
def annotate_txt(texts, device, bs=4, cfg=None):
    """Mean log-probability of each verbaliser's tokens as the start of the assistant's reply.
    bs texts per batch (bs x verbalisers sequences): 4 keeps the 7B model within 24 GB next to other jobs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = cfg or C.ANNOTATORS["TXT"]
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], dtype=torch.bfloat16).to(device).eval()
    verb_ids = [tok(v, add_special_tokens=False)["input_ids"] for v in cfg["verbalizers"]]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    scores = np.zeros((len(texts), len(verb_ids)), dtype=np.float64)
    for b in range(0, len(texts), bs):
        seqs, meta = [], []
        for j, text in enumerate(texts[b:b + bs]):
            msgs = [{"role": "user", "content": cfg["prompt"].format(text=text)}]
            prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            for k, v in enumerate(verb_ids):
                seqs.append(p_ids + v)
                meta.append((j, k, len(p_ids), len(v)))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):                      # right padding: positions of the prefix are unchanged
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        # hidden states only; the vocabulary projection is applied at the few positions that are scored
        h = model.model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
        for r, (j, k, lp, lv) in enumerate(meta):
            tgt = ids[r, lp:lp + lv].to(device)
            logp = torch.log_softmax(model.lm_head(h[r, lp - 1:lp - 1 + lv]).float(), -1)
            scores[b + j, k] = logp.gather(1, tgt[:, None]).mean().item()
        if (b // bs) % 500 == 0:
            print(f"annotate_txt {b}/{len(texts)}", flush=True)
    return scores.argmax(1), scores, f"{cfg['name']}@{_hf_revision(cfg['name'])}"


@torch.no_grad()
def annotate_aud(waves48, device, cfg=None, model=None):
    cfg = cfg or C.ANNOTATORS["AUD"]
    ckpt, fusion = cfg.get("ckpt", "630k-audioset-best.pt"), bool(cfg.get("fusion", False))
    if model is None:
        model = load_clap(ckpt, fusion, device, cfg.get("amodel", "HTSAT-tiny"))
    names = [c.replace("_", " ") for c in C.US8K_CLASSES]
    t = model.get_text_embedding([cfg["template"].format(c) for c in names], use_tensor=True).float()
    t = t / t.norm(dim=-1, keepdim=True)
    S = []
    for w in waves48:                                     # one clip at a time; the module repeat-pads to 10 s
        a = model.get_audio_embedding_from_data(x=np.asarray(w, dtype=np.float32)[None, :], use_tensor=False)
        a = torch.tensor(a, device=t.device).float()
        a = a / a.norm(dim=-1, keepdim=True)
        S.append((a @ t.T).cpu().numpy()[0])
    S = np.stack(S)
    return S.argmax(1), S, f"laion_clap:{ckpt[:-3]}:{_hf_revision('lukewys/laion_clap')}"


def load_clap(ckpt, fusion, device, amodel="HTSAT-tiny"):
    import laion_clap
    from huggingface_hub import hf_hub_download
    model = laion_clap.CLAP_Module(enable_fusion=fusion, device=device, amodel=amodel)
    # checkpoints loaded by hand because recent transformers versions no longer have the position_ids buffer
    # stored in them and torch >= 2.6 loads with weights_only=True
    path = hf_hub_download("lukewys/laion_clap", ckpt)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.endswith("position_ids")}
    res = model.model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys and all("position_ids" in k for k in res.missing_keys), res
    model.eval()
    return model


def run(ds, X_train, y_train, device):
    fn = {"IMG": annotate_img, "TXT": annotate_txt, "AUD": annotate_aud}[ds]
    yhat, scores, rev = fn(X_train, device)
    K = C.DATASETS[ds]["K"]
    r_F, T_F = N.annotator_stats(y_train, yhat, K)
    os.makedirs(C.NOISE, exist_ok=True)
    np.savez_compressed(os.path.join(C.NOISE, f"{ds}_F.npz"), yhat=yhat, scores=scores, revision=rev,
                        r_F=r_F, T_F=T_F)
    return yhat, r_F, T_F, rev
