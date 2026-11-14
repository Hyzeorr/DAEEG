import gc
import os
import pprint
import torch
import hydra
import time

from torch.amp import autocast
from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from datasets.caueeg_script import build_dataset_for_train
from run_train import check_device_env
from copy import deepcopy
from train.evaluate import check_accuracy_extended, check_accuracy_multicrop
from train.visualize import draw_roc_curve, draw_confusion, draw_class_wise_metrics
from models.utils import count_parameters


def compose_dataset(config):
    return build_dataset_for_train(config)

def load_pretrained_params(model, config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"],
    weights_only=False,) ###########

# ==================================================
    if config.get("input_norm") == "dataset":
        for k in ["signal_mean", "signal_std"]:
            if k not in config and k in ckpt["config"]:
                v = ckpt["config"][k]
                if torch.is_tensor(v):
                    v = v.detach().cpu()
                config[k] = v
    # ==================================================

    if ckpt["config"]["ddp"] == config["ddp"]:
        model.load_state_dict(ckpt["model_state"])
    elif ckpt["config"]["ddp"]:
        model_state_ddp = deepcopy(ckpt["model_state"])
        model_state = OrderedDict()
        for k, v in model_state_ddp.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            model_state[name] = v
        model.load_state_dict(model_state)
    else:
        model.module.load_state_dict(ckpt["model_state"])

def load_ckpt_and_patch_config(config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"),
                      map_location=config["device"], weights_only=False)

    # 1) in_channels 채우기
    if "in_channels" not in config or config["in_channels"] is None:
        candidates = ["input_proj.weight", "stem.0.weight", "conv1.weight", "patch_embed.proj.weight"]
        found = None
        for k in candidates:
            if k in ckpt["model_state"]:
                found = ckpt["model_state"][k].shape[1]; break
            kk = "module." + k
            if kk in ckpt["model_state"]:
                found = ckpt["model_state"][kk].shape[1]; break
        if found is None:
            raise KeyError("Cannot infer in_channels from checkpoint.")
        config["in_channels"] = int(found)

    # 2) dataset norm 통계도 채우기
    if config.get("input_norm") == "dataset":
        for k in ["signal_mean", "signal_std"]:
            if k not in config and k in ckpt["config"]:
                v = ckpt["config"][k]
                if torch.is_tensor(v):
                    v = v.detach().cpu()
                config[k] = v

    return ckpt

def patch_in_channels_from_ckpt(config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"),
                      map_location="cpu", weights_only=False)

    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    # DDP prefix 대응을 위해 helper
    def get_weight(key: str):
        if key in sd:
            return sd[key]
        k2 = "module." + key
        if k2 in sd:
            return sd[k2]
        return None

    # ✅ 1) in_channels 추정: “입력 채널 축이 들어가는 weight”를 후보로 탐색
    # - XEEGNet: temporal.0.conv.weight (in_ch=1), spatial.0.weight (in_ch=1)
    # - ResNet류: input_proj/stem/conv1 등
    candidates = [
        "input_proj.weight",
        "input_stage.0.weight",
        "conv_stage1.0.weight",
        "stem.0.weight",
        "conv1.weight",
        "patch_embed.proj.weight",

        # XEEGNet류
        "temporal.0.conv.weight",
        "temporal.0.weight",          # 구현마다 다를 수 있어서
        "spatial.0.weight",
        "separable.0.conv.weight",
    ]

    w = None
    used_key = None
    for k in candidates:
        w = get_weight(k)
        if w is not None:
            used_key = k
            break

    if w is None:
        # fallback: conv/proj 포함된 weight 중 하나 선택
        fallback = [k for k in sd.keys() if k.endswith("weight") and ("conv" in k or "proj" in k or "temporal" in k or "spatial" in k)]
        if not fallback:
            raise KeyError(f"Cannot infer in_channels. First 50 keys: {list(sd.keys())[:50]}")
        used_key = sorted(fallback)[0]
        w = sd[used_key]

    # 대부분 conv weight shape: (out, in, k) or (out, in, k, 1) or (out, in, ...)
    in_ch = int(w.shape[1])
    config["in_channels"] = in_ch
    print(f"[patch] in_channels <- {in_ch} (from '{used_key}' shape={tuple(w.shape)})")

    # ✅ 2) XEEGNet spatial 커널 높이(H)도 필요하면 같이 패치
    # checkpoint에 spatial.0.weight가 있으면 H=20 같은 값이 들어있음
    w_sp = get_weight("spatial.0.weight")
    if w_sp is not None and w_sp.ndim == 4:
        H = int(w_sp.shape[2])  # [out, in, H, W]
        # config에 존재하는 키만 업데이트 (프로젝트에 맞춰 확장 가능)
        for cand in ["n_channels", "eeg_channels", "C", "spatial_kernel"]:
            if cand in config:
                config[cand] = H
        if isinstance(config.get("model"), dict):
            for cand in ["n_channels", "eeg_channels", "C", "spatial_kernel"]:
                if cand in config["model"]:
                    config["model"][cand] = H
        print(f"[patch] xeeg spatial_kernel(H) <- {H} (from spatial.0.weight shape={tuple(w_sp.shape)})")

    return ckpt


def inference_script(
        config,
        model,
        test_loader,
        multicrop_test_loader,
        preprocess,
    ):
    model_state = deepcopy(model.state_dict())
    model.load_state_dict(model_state)
    print("Num_params :", count_parameters(model))

    model.eval()
    model.to(config["device"])
    
    # ===============================
    # Model-only latency measurement
    # ===============================
    device = config["device"]
    batch = next(iter(test_loader))
    x = batch["signal"][:1].to(device)

    age = batch.get("age", None)
    if age is None:
        # age가 없으면 0으로 더미 (B=1)
        age = torch.zeros((1, 1), device=device)
    else:
        age = age[:1].to(device)
        # shape이 (B,)면 (B,1)로 맞추기
        if age.ndim == 1:
            age = age.unsqueeze(1)

    # warmup / timing 에서
    _ = model(x, age)
    
    # warm-up
    with torch.no_grad():
        for _ in range(100):
           _ = model(x, age) 

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()

    # timing
    N = 300
    t0_lat = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = model(x, age)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    t1_lat = time.perf_counter()

    latency_ms = (t1_lat - t0_lat) / N * 1000
    print(f"[Timing] model-only latency (B=1): {latency_ms:.3f} ms")
    print(f"[Timing] model-only effective latency (8-crop): {latency_ms*8:.3f} ms")
    # ===============================
    # Model-only latency measurement
    # ===============================


    if torch.cuda.is_available() and "cuda" in str(config["device"]):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    test_result = check_accuracy_extended(
        model=model,
        loader=test_loader,
        preprocess=preprocess,
        config=config,
        # repeat=config.get("test_accuracy_repeat", 30),
        repeat=config.get("test_accuracy_repeat", 1),
    )

    if torch.cuda.is_available() and "cuda" in str(config["device"]):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    num_test = len(test_loader.dataset)
    print(f"[Timing] check_accuracy_extended total: {elapsed:.3f} s")
    print(f"[Timing] end-to-end throughput: {num_test/elapsed:.2f} samples/s")

    test_acc, score, target, test_confusion, throughput  = test_result

    if torch.cuda.is_available() and "cuda" in str(config["device"]):
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    # multicrop_test_acc = check_accuracy_multicrop(
    #     model=model,
    #     loader=multicrop_test_loader,
    #     preprocess=preprocess,
    #     config=config,
    #     # repeat=config.get("test_accuracy_repeat", 30),
    #     repeat=config.get("test_accuracy_repeat", 1),
    # )

    if torch.cuda.is_available() and "cuda" in str(config["device"]):
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    elapsed_mc = t4 - t3
    num_mc = len(multicrop_test_loader.dataset)
    print(f"[Timing] check_accuracy_multicrop total: {elapsed_mc:.3f} s")
    print(f"[Timing] multicrop end-to-end throughput: {num_mc/elapsed_mc:.2f} samples/s")

    pprint.pprint(
        {
            f"Test Accuracy": test_acc,
            "Confusion Matrix (Array)": test_confusion,
            #"Multi-Crop Test Accuracy": multicrop_test_acc,

        }
    )
    print(f"\n{'*' * 92}\n")

    # draw_roc_curve(
    #     score,
    #     target,
    #     config["class_label_to_name"],
    #     use_wandb=config["use_wandb"],
    # )
    # draw_confusion(
    #     test_confusion,
    #     config["class_label_to_name"],
    #     normalize=True,
    #     use_wandb=config["use_wandb"],
    # )
    # draw_class_wise_metrics(
    #     test_confusion,
    #     config['class_label_to_name'],
    #     use_wandb=config["use_wandb"]
    # )

# def run_inference(config):
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()

#     ckpt = patch_in_channels_from_ckpt(config)
#     # model = generate_model(config)
#     model = hydra.utils.instantiate(config)
#     load_pretrained_params(model, config)
    
#     _, _, test_loader, multicrop_test_loader = compose_dataset(config)

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t0 = time.perf_counter()
    
#     inference_script(config, model, test_loader, multicrop_test_loader, config["preprocess_test"])
    
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t1 = time.perf_counter()

#     elapsed = t1 - t0
#     num_samples = len(test_loader.dataset)

#     print(f"[Timing][E2E] total inference time: {elapsed:.3f} s")
#     print(f"[Timing][E2E] throughput: {num_samples / elapsed:.2f} samples/s")

def run_inference(config, cfg_model):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    ckpt = patch_in_channels_from_ckpt(config)
    cfg_model.in_channels = config["in_channels"]
    cfg_model.out_dims = config.get("out_dims", cfg_model.out_dims)

    model = hydra.utils.instantiate(cfg_model)
    load_pretrained_params(model, config)
    
    _, _, test_loader, multicrop_test_loader = compose_dataset(config)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    inference_script(config, model, test_loader, multicrop_test_loader, config["preprocess_test"])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    num_samples = len(test_loader.dataset)

    print(f"[Timing][E2E] total inference time: {elapsed:.3f} s")
    print(f"[Timing][E2E] throughput: {num_samples / elapsed:.2f} samples/s")

@hydra.main(config_path="config", config_name="test")
def my_app(cfg: DictConfig) -> None:
    config = {
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.train),
        **OmegaConf.to_container(cfg.model),
        "cwd": HydraConfig.get().runtime.cwd,
    }

    config["out_dims"] = 3  # 예시

    # check the workstation environment and update some configurations
    check_device_env(config)

    run_inference(config=config, cfg_model=cfg.model)

if __name__ == "__main__":
    print("I am starting now")
    my_app()
