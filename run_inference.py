import gc
import os
import pprint
import torch
import hydra
import time
import numpy as np

from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf, open_dict
from datasets.caueeg_script import build_dataset_for_train
from run_train import check_device_env
from copy import deepcopy
from train.evaluate import check_accuracy_extended, check_accuracy_multicrop
from train.visualize import draw_roc_curve, draw_confusion, draw_class_wise_metrics
from models.utils import count_parameters

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

########################################################
# inference time measure
########################################################
def _sync(device):
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()

def measure_model_only_latency_real_batch(
    model,
    loader,
    device,
    preprocess=None,   # None이면 preprocess 제외
    warmup=50,
    iters=200,
    use_amp=False,
):
    model.eval()
    model.to(device)

    batch = next(iter(loader))   # dict
    if preprocess is not None:
        batch = preprocess(batch)

    # ✅ 너 데이터셋 dict 키에 맞게 추출
    x = batch["signal"]          # (B,C,T)
    age = batch["age"]           # (B,) or (B,1) etc.

    # batch=1
    x = x[:1].to(device, non_blocking=True)
    age = age[:1].to(device, non_blocking=True)

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            if use_amp and ("cuda" in str(device)):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x, age)
            else:
                _ = model(x, age)
        _sync(device)

        # measure
        for _ in range(iters):
            starter.record()
            if use_amp and ("cuda" in str(device)):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x, age)
            else:
                _ = model(x, age)
            ender.record()
            _sync(device)
            times_ms.append(starter.elapsed_time(ender))

    times_ms = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_sps": float(1000.0 / times_ms.mean()) if times_ms.mean() > 0 else 0.0,
        "n": int(len(times_ms)),
    }
########################################################
########################################################

def compose_dataset(config):
    return build_dataset_for_train(config)

# def load_pretrained_params(model, config):
#     lp = config.get("load_pretrained") or config.get("train", {}).get("load_pretrained")
#     if not lp:
#         raise KeyError("Missing load_pretrained. Set it in config/train/test yaml or run with train.load_pretrained=exp001")

#     save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{lp}/")
#     ckpt_path = os.path.join(save_path, "checkpoint.pt")
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

#     ckpt = torch.load(ckpt_path, map_location=config["device"])

#     # ckpt 로드 직후
#     config["class_label_to_name"] = ckpt["config"].get(
#         "class_label_to_name",
#         config.get("class_label_to_name", ["Normal", "MCI", "Dementia"])
#     )

#     # 학습 때 쓰던 정규화/나이 통계를 그대로 쓰도록 (있으면 덮어쓰기)
#     for k in ["age_mean", "age_std", "signal_mean", "signal_std", "input_norm", "EKG"]:
#         if k in ckpt["config"]:
#             config[k] = ckpt["config"][k]

#     if ckpt["config"]["ddp"] == config["ddp"]:
#         model.load_state_dict(ckpt["model_state"])
#     elif ckpt["config"]["ddp"]:
#         model_state_ddp = deepcopy(ckpt["model_state"])
#         model_state = OrderedDict()
#         for k, v in model_state_ddp.items():
#             model_state[k[7:]] = v  # remove 'module.'
#         model.load_state_dict(model_state)
#     else:
#         model.module.load_state_dict(ckpt["model_state"])

def _remap_legacy_mamba_fc_keys(model_state, target_model):
    """Map legacy flat `fc.*` keys to current Mamba1D `fc_stage`/nested `fc.*` keys."""
    target_keys = set(target_model.state_dict().keys())
    needs_staged_fc = any(k.startswith("fc_stage.") for k in target_keys)
    has_legacy_flat_fc = "fc.0.weight" in model_state and "fc.1.running_mean" in model_state
    if not (needs_staged_fc and has_legacy_flat_fc):
        return model_state

    remapped = OrderedDict(model_state)
    legacy_to_new = {
        "fc.0.weight": ["fc.0.0.weight", "fc_stage.0.0.weight"],
        "fc.1.weight": ["fc.0.1.weight", "fc_stage.0.1.weight"],
        "fc.1.bias": ["fc.0.1.bias", "fc_stage.0.1.bias"],
        "fc.1.running_mean": ["fc.0.1.running_mean", "fc_stage.0.1.running_mean"],
        "fc.1.running_var": ["fc.0.1.running_var", "fc_stage.0.1.running_var"],
        "fc.1.num_batches_tracked": ["fc.0.1.num_batches_tracked", "fc_stage.0.1.num_batches_tracked"],
        "fc.4.weight": ["fc.1.weight", "fc_stage.1.weight"],
        "fc.4.bias": ["fc.1.bias", "fc_stage.1.bias"],
    }

    for legacy_k, new_keys in legacy_to_new.items():
        if legacy_k in model_state:
            v = model_state[legacy_k]
            for new_k in new_keys:
                if new_k in target_keys:
                    remapped[new_k] = v

    for legacy_k in legacy_to_new:
        remapped.pop(legacy_k, None)

    return remapped


def _drop_mismatched_keys(model_state, target_model):
    target_state = target_model.state_dict()
    filtered = OrderedDict()
    dropped = []
    for k, v in model_state.items():
        if k not in target_state:
            filtered[k] = v
            continue
        if getattr(v, "shape", None) != target_state[k].shape:
            dropped.append((k, tuple(v.shape), tuple(target_state[k].shape)))
            continue
        filtered[k] = v
    return filtered, dropped


def load_pretrained_params(model, config, ckpt):
    if ckpt["config"]["ddp"] == config["ddp"]:
        target_model = model
        model_state = ckpt["model_state"]
    elif ckpt["config"]["ddp"]:
        target_model = model
        model_state = OrderedDict((k[7:], v) for k, v in ckpt["model_state"].items())
    else:
        target_model = model.module
        model_state = ckpt["model_state"]

    model_state = _remap_legacy_mamba_fc_keys(model_state, target_model)
    model_state, dropped = _drop_mismatched_keys(model_state, target_model)
    if dropped:
        print("[ckpt] dropped shape-mismatched keys:")
        for k, src_shape, dst_shape in dropped:
            print(f"  - {k}: ckpt{src_shape} != model{dst_shape}")

    missing, unexpected = target_model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[ckpt] missing keys after load ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[ckpt] unexpected keys after load ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

def load_ckpt_config_only(config):
    lp = config.get("load_pretrained") or config.get("train", {}).get("load_pretrained")
    if not lp:
        raise KeyError("Missing load_pretrained. Set train.load_pretrained=exp001")

    save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{lp}/")
    ckpt_path = os.path.join(save_path, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 평가/시각화 라벨
    config["class_label_to_name"] = ckpt["config"].get(
        "class_label_to_name",
        config.get("class_label_to_name", ["Normal", "MCI", "Dementia"])
    )

    # 학습 때 통계/정규화 설정을 그대로 사용
    for k in ["age_mean", "age_std", "signal_mean", "signal_std", "input_norm", "EKG"]:
        if k in ckpt["config"]:
            config[k] = ckpt["config"][k]

    return ckpt

def load_ckpt_and_merge_config(config):
    lp = config.get("load_pretrained") or config.get("train", {}).get("load_pretrained")
    save_path = os.path.join(config.get("cwd",""), f"local/checkpoint/{lp}/")
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location="cpu")

    # 중요: ckpt 기준으로 세팅
    for k in ["class_label_to_name", "input_norm", "EKG", "photic", "task"]:
        if k in ckpt["config"]:
            config[k] = ckpt["config"][k]

    # --- 여기부터가 교체되는 부분 ---

    def _to_scalar(x, default=None):
        if torch.is_tensor(x):
            x = x.detach().cpu()
            return float(x.item()) if x.numel() == 1 else float(x.flatten()[0].item())
        if isinstance(x, (list, tuple)):
            return float(x[0]) if len(x) > 0 else default
        if isinstance(x, (int, float)):
            return float(x)
        return default

    # ✅ signal_mean/std는 채널별이라 list로
    for k in ["signal_mean", "signal_std"]:
        if k in ckpt["config"]:
            v = ckpt["config"][k]
            if torch.is_tensor(v):
                v = v.detach().cpu().flatten().tolist()
            config[k] = v

    # ✅ age_mean/std는 반드시 scalar float
    for k in ["age_mean", "age_std"]:
        if k in ckpt["config"]:
            config[k] = _to_scalar(
                ckpt["config"][k],
                default=(0.0 if k == "age_mean" else 1.0)
            )

     # -----------------------------
    # ✅ in_channels 추정: 안전 버전
    # -----------------------------
    state = ckpt.get("model_state", {})
    # 1) ckpt config에 in_channels가 있으면 그걸 우선 사용
    if "in_channels" in ckpt.get("config", {}):
        config["in_channels"] = int(ckpt["config"]["in_channels"])
    else:
        # 2) model_state에서 'input_proj.weight'로 끝나는 키를 찾기
        w = None
        for k, v in state.items():
            if k.endswith("input_proj.weight"):
                w = v
                break
        # 3) 그래도 없으면, 신호 입력을 받는 첫 conv/linear weight 후보를 찾기
        if w is None:
            for k, v in state.items():
                if k.endswith("weight") and hasattr(v, "dim") and v.dim() >= 2:
                    # 너무 광범위하면 위험하니, 이름에 conv/proj/stem 같은 힌트가 있는 것만
                    lk = k.lower()
                    if any(t in lk for t in ["conv", "proj", "stem", "input"]):
                        w = v
                        break

        # 4) 최후: 지금 config에 있는 값 유지(혹은 signal_mean 길이로 추정)
        if w is not None:
            config["in_channels"] = int(w.shape[1])
        else:
            if "signal_mean" in config and isinstance(config["signal_mean"], list):
                config["in_channels"] = len(config["signal_mean"])
            else:
                config.setdefault("in_channels", 19)  # fallback

    # model hyperparameters: prefer ckpt values for shape-sensitive fields
    for k in [
        "use_age",
        "fc_stages",
        "dropout",
        "d_model",
        "n_layers",
        "d_state",
        "d_conv",
        "expand",
        "multi_scale_conv",
        "multi_scale_kernels",
        "channel_attention",
        "se_reduction",
        "adapter_hidden",
        "lora_alpha",
        "lora_dropout",
        "seq_length",
    ]:
        if k in ckpt["config"]:
            config[k] = ckpt["config"][k]

    # out_dims
    config["out_dims"] = ckpt["config"].get("out_dims", len(config["class_label_to_name"]))
    return ckpt


#######################################################################
#########################   t-SNE 유틸 함수   ###########################
#######################################################################
def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.inference_mode()
def export_tsne_from_inference(
    model,
    loader,
    preprocess,
    device,
    class_names,
    out_path="tsne.png",
    max_per_class=1000,
    perplexity=40,
    seed=0,
):
    """
    - model(x, age) 출력(보통 (B,D))을 feature로 보고 t-SNE 수행
    - loader batch는 dict이며 keys: 'signal', 'age', 'class_label'(또는 'label'/'target') 가정
    """
    seed_all(seed)
    model.eval()
    model.to(device)

    n_classes = len(class_names)
    per_class_cnt = [0] * n_classes

    feats_list = []
    y_list = []

    for batch in loader:
        # preprocess 적용 (너 코드에서 accuracy 평가할 때 쓰는 것과 동일하게)
        if preprocess is not None:
            batch = preprocess(batch)

        x = batch["signal"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)

        # label key가 다를 수 있어 안전 처리
        if "class_label" in batch:
            y = batch["class_label"]
        elif "label" in batch:
            y = batch["label"]
        elif "target" in batch:
            y = batch["target"]
        else:
            raise KeyError(
                f"batch에 라벨 키가 필요합니다. 지원 키: 'class_label'/'label'/'target', 실제 키: {list(batch.keys())}"
            )
        y = y.detach().cpu().numpy()

        out = model(x, age)  # (B,D) 라고 가정(너 말대로 forward가 embedding 리턴)
        out = out.detach().cpu().numpy()

        keep_idx = []
        for i, yi in enumerate(y):
            yi = int(yi)
            if 0 <= yi < n_classes and per_class_cnt[yi] < max_per_class:
                per_class_cnt[yi] += 1
                keep_idx.append(i)

        if keep_idx:
            feats_list.append(out[keep_idx])
            y_list.append(y[keep_idx])

        if all(c >= max_per_class for c in per_class_cnt):
            break

    feats = np.concatenate(feats_list, axis=0)
    ys = np.concatenate(y_list, axis=0)

    # (선택) 출력 차원이 너무 작으면(예: logits 2~3차원) t-SNE 의미가 약할 수 있음
    if feats.shape[1] <= 3:
        print(f"[WARN] feature dim={feats.shape[1]} (logits일 가능성). 그래도 진행은 합니다.")

    feats = StandardScaler().fit_transform(feats)
    z = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(feats)

    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    for c in np.unique(ys):
        idx = (ys == c)
        name = class_names[int(c)] if int(c) < len(class_names) else f"class {int(c)}"
        ax.scatter(z[idx, 0], z[idx, 1], z[idx, 2], s=6, alpha=0.7, label=name)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(markerscale=2, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[t-SNE] saved: {out_path} | N={len(ys)} | dim={feats.shape[1]} | per_class={per_class_cnt}")

def inference_script(config, model, test_loader, multicrop_test_loader, preprocess):
    print("Num_params :", count_parameters(model))
    print(f"[Info] #test samples (dataset): {len(test_loader.dataset)}")
    print(f"[Info] #test batches (loader): {len(test_loader)}")
    print(f"[Info] test batch_size (first batch): {next(iter(test_loader))['signal'].shape[0]}")
    print(f"[Info] #multicrop samples (dataset): {len(multicrop_test_loader.dataset)}")
    print(f"[Info] #multicrop batches (loader): {len(multicrop_test_loader)}")

    model.eval()
    model.to(config["device"])

    test_acc, score, target, test_confusion, throughput = check_accuracy_extended(
        model=model,
        loader=test_loader,
        preprocess=preprocess,
        config=config,
        repeat=config.get("test_accuracy_repeat", 1),
    )

    # multicrop_test_acc = check_accuracy_multicrop(
    #     model=model,
    #     loader=multicrop_test_loader,
    #     preprocess=preprocess,
    #     config=config,
    #     repeat=config.get("test_accuracy_repeat", 1),
    # )

    pprint.pprint(
        {
            "Test Accuracy": test_acc,
            "Confusion Matrix (Array)": test_confusion,
            #"Multi-Crop Test Accuracy": multicrop_test_acc,
        }
    )
    print(f"\n{'*' * 92}\n")

    # draw_roc_curve(score, target, config["class_label_to_name"], use_wandb=config["use_wandb"])
    # draw_confusion(test_confusion, config["class_label_to_name"], normalize=True, use_wandb=config["use_wandb"])
    # draw_class_wise_metrics(test_confusion, config["class_label_to_name"], use_wandb=config["use_wandb"])


# def run_inference(config, model, preprocess_test):
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

#     config["skip_signal_statistics"] = True
#     # config["skip_signal_statistics"] = False
#     config.setdefault("age_mean", 0.0)
#     config.setdefault("age_std", 1.0)

#     # 데이터/모델과 일치하도록 설정 (필요시 CLI로 override 권장)
#     config.setdefault("in_channels", 19)

#     _, _, test_loader, multicrop_test_loader = compose_dataset(config)

#     load_pretrained_params(model, config)
#     inference_script(config, model, test_loader, multicrop_test_loader, preprocess_test)

def run_inference(config, model):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    config["skip_signal_statistics"] = True
    ckpt = load_ckpt_and_merge_config(config)

    print(type(config["age_mean"]), config["age_mean"])
    print(type(config["age_std"]), config["age_std"])

    _, _, test_loader, multicrop_test_loader = compose_dataset(config)
    preprocess_test = config["preprocess_test"]
    def patch_norm_for_bct(preprocess):

        patched = 0
        for m in preprocess.modules():
            if hasattr(m, "mean") and hasattr(m, "std_eps"):
                mean = getattr(m, "mean")
                std  = getattr(m, "std_eps")

                if torch.is_tensor(mean) and mean.dim() == 1:
                    C = mean.numel()
                    m.mean = mean.view(1, C, 1)

                    if torch.is_tensor(std) and std.dim() == 1 and std.numel() == C:
                        m.std_eps = std.view(1, C, 1)

                    patched += 1
        print(f"[preprocess patch] normalized modules patched: {patched}")
        return preprocess

    patch_norm_for_bct(preprocess_test)

    # 모델 state 로드
    load_pretrained_params(model, config, ckpt)

    # 시간 측정 시작 
    # (A) model-only (preprocess 제외)
    lat_no_pp = measure_model_only_latency_real_batch(
        model=model,
        loader=test_loader,
        device=config["device"],
        preprocess=None,
        warmup=50,
        iters=200,
        use_amp=False,
    )
    print("[Timing][Model-only][no preprocess]", lat_no_pp)

    # (B) model-only (preprocess 포함) - 선택/권장
    lat_with_pp = measure_model_only_latency_real_batch(
        model=model,
        loader=test_loader,
        device=config["device"],
        preprocess=preprocess_test,
        warmup=50,
        iters=200,
        use_amp=False,
    )
    print("[Timing][Model-only][with preprocess]", lat_with_pp)
    #   시간 측정 종료 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    inference_script(config, model, test_loader, multicrop_test_loader, preprocess_test)
    
       # === 추가: t-SNE export (inference-only) ===
    export_tsne_from_inference(
        model=model,
        loader=test_loader,
        preprocess=preprocess_test,
        device=config["device"],
        class_names=config["class_label_to_name"],
        out_path=os.path.join(config.get("cwd", "."), "tsne_test.png"),
        max_per_class=1000,
        perplexity=40,
        seed=0,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    num_samples = len(test_loader.dataset)

    print(f"[Timing][E2E] total inference time: {elapsed:.3f} s")
    print(f"[Timing][E2E] throughput: {num_samples / elapsed:.2f} samples/s")

def identity_preprocess(sample_batched):
    return sample_batched

@hydra.main(config_path="config", config_name="test")
def my_app(cfg: DictConfig) -> None:
    # 1) config dict (데이터로더/평가 함수들이 dict를 기대하므로)
    config = {
        **OmegaConf.to_container(cfg.data, resolve=True),
        **OmegaConf.to_container(cfg.train, resolve=True),
        **OmegaConf.to_container(cfg.model, resolve=True),
        "cwd": HydraConfig.get().runtime.cwd,
    }

    # 2) device 등 런타임 환경 보정
    check_device_env(config)

    ckpt = load_ckpt_and_merge_config(config)

    # ✅ cfg.model도 ckpt 기준으로 수정 후 instantiate
    with open_dict(cfg):
        for k in [
            "in_channels",
            "out_dims",
            "seq_length",
            "use_age",
            "fc_stages",
            "dropout",
            "d_model",
            "n_layers",
            "d_state",
            "d_conv",
            "expand",
            "multi_scale_conv",
            "multi_scale_kernels",
            "channel_attention",
            "se_reduction",
            "adapter_hidden",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
        ]:
            if k in config:
                setattr(cfg.model, k, config[k])

    model = hydra.utils.instantiate(cfg.model)

    run_inference(config=config, model=model)

if __name__ == "__main__":
    print("I am starting now")
    my_app()
