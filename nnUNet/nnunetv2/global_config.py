import os

# Centralized root for externally stored checkpoints.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)

PRE_TRAINED_PATH = os.path.join(ROOT_DIR, "pre_trained_models")

DINOV3_VITB16_PRETRAIN_CKPT = os.path.join(
    PRE_TRAINED_PATH,
    "dinov3_vitb16_pretrain_lvd1689m.pth",
)

DINOV3_VITL16_PRETRAIN_CKPT = os.path.join(
    PRE_TRAINED_PATH,
    "dinov3_vitl16_pretrain_lvd1689m.pth",
)

DINO_V3_DEFAULT_PRETRAINED = DINOV3_VITL16_PRETRAIN_CKPT

MEDDINOV3_CT_CKPT = os.path.join(
    PRE_TRAINED_PATH,
    "meddino_checkpoint.pth",
)
