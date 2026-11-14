import gc
import os
import pprint
import torch
import hydra
import time

import torch.nn as nn
from torch.amp import autocast
from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf, open_dict
from datasets.caueeg_script import build_dataset_for_train
from run_train import check_device_env
from copy import deepcopy
from train.evaluate import check_accuracy_extended, check_accuracy_multicrop
from train.visualize import draw_roc_curve, draw_confusion, draw_class_wise_metrics
from models.utils import count_parameters


def compose_dataset(config):
    return build_dataset_for_train(config)

def load_pretrained_params(model, config, ckpt):
    sd = ckpt["model_state"]

    # ddp mismatch 처리
    if ckpt["config"]["ddp"] == config["ddp"]:
        model.load_state_dict(sd)
    elif ckpt["config"]["ddp"]:
        sd2 = OrderedDict((k[7:], v) for k, v in sd.items())  # remove module.
        model.load_state_dict(sd2)
    else:
        model.module.load_state_dict(sd)
 
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

def load_ckpt(config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt_path = os.path.join(save_path, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt

def patch_xeeg_config_from_ckpt(config, cfg_model, ckpt):
    sd = ckpt["model_state"]

    # ddp prefix 대응
    def get_w(k):
        return sd.get(k, sd.get("module." + k))

    # ✅ EEG 채널 수 = spatial 커널 높이(H)
    w_sp = get_w("spatial.0.weight")
    if w_sp is None:
        raise KeyError("XEEGNet ckpt must have spatial.0.weight")

    eeg_C = int(w_sp.shape[2])  # ex) 20
    config["in_channels"] = eeg_C

    # ✅ out_dims / 클래스 수도 ckpt config가 있으면 그대로
    config["out_dims"] = int(ckpt["config"].get("out_dims", config.get("out_dims", 3)))

    # ✅ 모델 하이퍼파라미터도 ckpt config로 통일 (있으면)
    for k in ["F1", "D", "F2", "kernel_length", "pool_size_1", "pool_size_2", "dropout", "use_bn", "return_feature", "seq_length", "n_classes"]:
        if k in ckpt["config"]:
            config[k] = ckpt["config"][k]

    if "signal_mean" in ckpt["config"] and "signal_std" in ckpt["config"]:
        config["signal_mean"] = ckpt["config"]["signal_mean"]
        config["signal_std"]  = ckpt["config"]["signal_std"]

    if "age_mean" in ckpt["config"] and "age_std" in ckpt["config"]:
        config["age_mean"] = float(ckpt["config"]["age_mean"])
        config["age_std"]  = float(ckpt["config"]["age_std"])

    # ✅ dataset 통계 다시 계산하지 않도록
    config["input_norm"] = ckpt["config"].get("input_norm", "none")
    config["skip_signal_statistics"] = True

    if torch.is_tensor(config["signal_mean"]):
        config["signal_mean"] = config["signal_mean"].detach().cpu().flatten().tolist()
    if torch.is_tensor(config["signal_std"]):
        config["signal_std"] = config["signal_std"].detach().cpu().flatten().tolist()


    # ✅ cfg.model에도 반영 (instantiate 전에!)
    with open_dict(cfg_model):
        cfg_model.in_channels = eeg_C
        cfg_model.seq_length = int(config["seq_length"])
        cfg_model.out_dims = int(config["out_dims"])
        cfg_model.n_classes = int(config["out_dims"])  # 너 모델 init이 n_classes로 받음
        cfg_model.F1 = int(config.get("F1", 8))
        cfg_model.D  = int(config.get("D", 2))
        cfg_model.F2 = int(config.get("F2", 16))
        cfg_model.kernel_length = int(config.get("kernel_length", 64))
        cfg_model.pool_size_1 = int(config.get("pool_size_1", 4))
        cfg_model.pool_size_2 = int(config.get("pool_size_2", 8))
        cfg_model.dropout = float(config.get("dropout", 0.25))
        cfg_model.use_bn = bool(config.get("use_bn", True))
        cfg_model.return_feature = bool(config.get("return_feature", False))

    print(f"[xeeg patch] in_channels={eeg_C}, out_dims={config['out_dims']}, seq_length={config['seq_length']}")

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

def run_inference(config, cfg_model):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ckpt = load_ckpt(config)

    # ✅ xeegnet 전용 patch
    patch_xeeg_config_from_ckpt(config, cfg_model, ckpt)

    model = hydra.utils.instantiate(cfg_model)
    load_pretrained_params(model, config, ckpt)

    for k in ["signal_mean", "signal_std", "age_mean", "age_std", "input_norm"]:
        if k in ckpt["config"]:
            v = ckpt["config"][k]
            if torch.is_tensor(v):
                v = v.detach().cpu().flatten().tolist() if k in ["signal_mean","signal_std"] else float(v.item())
            config[k] = v

    config["skip_signal_statistics"] = True
    config["input_norm"] = "ckpt"   # 혹은 "dataset"이 아니라 ckpt 통계를 쓰게

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