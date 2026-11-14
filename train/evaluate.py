import numpy as np
import torch
import torch.nn.functional as F

# __all__ = []


@torch.no_grad()
def compute_embedding(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]
    output = model.compute_feature_embedding(x, age, target_from_last=1)

    return output


@torch.no_grad()
def estimate_score(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]
    output = model(x, age)

    if config["criterion"] == "cross-entropy":
        score = F.softmax(output, dim=1)
    elif config["criterion"] == "multi-bce":
        score = torch.sigmoid(output)
    elif config["criterion"] == "svm":
        score = output
    else:
        raise ValueError(f"estimate_score(): cannot parse config['criterion']={config['criterion']}.")
    return score


def calculate_confusion_matrix(pred, target, num_classes):
    indices = target.to(torch.int64) * num_classes + pred.to(torch.int64)
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu().numpy()


def _merge_multicrop_scores(score, target, crops):
    if score.size(0) % crops != 0:
        raise ValueError(
            f"Multi-crop batch size={score.size(0)} is not divisible by test_crop_multiple={crops}."
        )
    return score.reshape(-1, crops, score.size(1)).mean(dim=1), target[::crops]


def calculate_class_wise_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    accuracy = np.zeros((n_classes,))
    sensitivity = np.zeros((n_classes,))
    specificity = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp

        accuracy[c] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[c] = tp / (tp + fn)
        specificity[c] = tn / (fp + tn)
        precision[c] = tp / (tp + fp)
        recall[c] = sensitivity[c]
    f1_score = 2 * precision * recall / (precision + recall)

    class_wise_metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
    }  # 'Recall': recall is same with sensitivity
    return class_wise_metrics


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score_chunks = []
    target_chunks = []

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            score_chunks.append(s.cpu().numpy())
            target_chunks.append(y.cpu().numpy())

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time
    score = np.concatenate(score_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)

    return accuracy, score, target, confusion_matrix, throughput


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            s, y = _merge_multicrop_scores(s, y, config["test_crop_multiple"])

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score_chunks = []
    target_chunks = []

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            s, y = _merge_multicrop_scores(s, y, config["test_crop_multiple"])

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            # classification score for drawing ROC curve
            score_chunks.append(s.cpu().numpy())
            target_chunks.append(y.cpu().numpy())

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time
    score = np.concatenate(score_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)

    return accuracy, score, target, confusion_matrix, throughput
