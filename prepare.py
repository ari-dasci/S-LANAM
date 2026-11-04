"""Downloads datasets and model weights into RND_DATA and the Hugging Face cache.

Usage:
  python prepare.py --what TAB,TS,GRA,TXT,AUD,IMG --models distilbert,e5,panns,qwen,clap,clip
  python prepare.py --what CIFAR10N,TAB2,IMG2,TXT2,TS2,GRA2,AUD2          # round 2
"""
import argparse
import os
import shutil
import sys
import urllib.request

from rnd import config as C
from rnd import data as D


def fetch(url, dest):
    if os.path.exists(dest):
        print("present", dest)
        return
    print("downloading", url, "->", dest, flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    os.replace(tmp, dest)


def data(what):
    if "TAB" in what:
        fetch(D.URLS["TAB"], os.path.join(D.ddir("TAB"), "letter+recognition.zip"))
        D.load_tab()
    if "TS" in what:
        fetch(D.URLS["TS"], os.path.join(D.ddir("TS"), "human+activity+recognition+using+smartphones.zip"))
        D.load_ts()
    if "IMG" in what:
        from torchvision.datasets import CIFAR10
        CIFAR10(D.ddir("IMG"), train=True, download=True)
        CIFAR10(D.ddir("IMG"), train=False, download=True)
    if "TXT" in what:
        D.load_txt()
    if "GRA" in what:
        D.load_gra()
    if "AUD" in what:
        fetch(D.URLS["AUD"], os.path.join(D.ddir("AUD"), "UrbanSound8K.tar.gz"))
        D.load_aud()
    # round 2
    if "CIFAR10N" in what:
        fetch(D.URLS["CIFAR10N"], os.path.join(D.ddir("IMG"), "CIFAR-10_human.pt"))
    if "TAB2" in what:
        fetch(D.URLS["TAB2"], os.path.join(D.ddir("TAB2"), "covertype.zip"))
        D.load_tab2()
    if "IMG2" in what:
        from torchvision.datasets import CIFAR100
        CIFAR100(D.ddir("IMG2"), train=True, download=True)
        CIFAR100(D.ddir("IMG2"), train=False, download=True)
    if "TXT2" in what:
        D.load_txt2()
    if "TS2" in what:
        fetch(D.URLS["TS2"], os.path.join(D.ddir("TS2"), "ElectricDevices.zip"))
        D.load_ts2()
    if "GRA2" in what:
        D.load_gra2()
    if "AUD2" in what:
        fetch(D.URLS["AUD2"], os.path.join(D.ddir("AUD2"), "speech_commands_v0.02.tar.gz"))
        D.load_aud2()


def models(which):
    from huggingface_hub import snapshot_download, hf_hub_download
    if "distilbert" in which:
        snapshot_download("distilbert-base-uncased", allow_patterns=["*.json", "*.txt", "*.safetensors"])
    if "e5" in which:
        snapshot_download("intfloat/e5-large-v2", allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*",
                                                                 "modules.json", "sentence_*"])
    if "qwen" in which:
        snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt",
                                                                      "tokenizer*"])
    if "clap" in which:
        hf_hub_download("lukewys/laion_clap", "630k-audioset-best.pt")
        snapshot_download("roberta-base", allow_patterns=["*.json", "*.txt", "*.safetensors"])
    if "panns" in which:
        d = os.path.join(os.path.expanduser("~"), "panns_data")
        os.makedirs(d, exist_ok=True)
        fetch("https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
              os.path.join(d, "Cnn14_mAP=0.431.pth"))
        fetch("https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv",
              os.path.join(d, "class_labels_indices.csv"))
    if "clip" in which:
        import open_clip
        open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
        open_clip.create_model_and_transforms("RN50", pretrained="openai", force_quick_gelu=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--what", default="")
    p.add_argument("--models", default="")
    a = p.parse_args()
    data(a.what.split(","))
    models(a.models.split(","))
    print("prepare: done")
    sys.exit(0)
