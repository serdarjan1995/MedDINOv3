import shutil

import huggingface_hub
import os

import torch
from safetensors.torch import load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def download_from_hf():
    meddino_pretrained_path = os.path.join(BASE_DIR, "meddino_checkpoint.pth")
    if not os.path.exists(meddino_pretrained_path):
        meddino_checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="ricklisz123/MedDINOv3-ViTB-16-CT-3M",
            filename="model.pth",
            local_dir=BASE_DIR,
            token=HF_TOKEN
        )
        custom_path = os.path.join(BASE_DIR, "meddino_checkpoint.pth")
        shutil.move(meddino_checkpoint_path, custom_path)
        print(f"Downloaded MedDINOv3-ViTB-16-CT-3M checkpoint to {custom_path}")

    dino_filename = "dinov3_vitl16_pretrain_lvd1689m.pth"
    dino_pth_path = os.path.join(BASE_DIR, dino_filename)

    if not os.path.exists(dino_pth_path):
        safetensors_file = os.path.join(BASE_DIR, "model.safetensors")
        if os.path.exists(safetensors_file):
            os.remove(safetensors_file)
        dinov3_pretrained_path = huggingface_hub.hf_hub_download(
            repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
            filename="model.safetensors",
            local_dir=BASE_DIR,
            token=HF_TOKEN
        )

        # Load safetensors
        state_dict = load_file(safetensors_file)

        # Save as .pth
        torch.save(state_dict, dino_pth_path)
        print(f"Downloaded dinov3-vitl16-pretrain-lvd1689m checkpoint to {dino_pth_path}")
        os.remove(safetensors_file)


if __name__ == "__main__":
    try:
        download_from_hf()
    except huggingface_hub.errors.GatedRepoError:
        print("Please use HF_TOKEN to download the pre-trained models. set env HF_TOKEN=<your_hf_token>")
