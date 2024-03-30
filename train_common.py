import numpy as np
from sklearn import metrics 


def find_optimal_thresholds(targets: np.ndarray, outputs: np.ndarray, metric_func, candidate_thresholds: list[float] | np.ndarray, num_labels: int):
    """
    Helper function to find optimal thresholds for each label in the multi-label classification task.
    """    
    optimal_thresholds = []

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    if not isinstance(outputs, np.ndarray):
        outputs = np.array(outputs)
    
    for label_idx in range(num_labels):
        best_threshold = 0
        best_metric = float('-inf')
        
        for threshold in candidate_thresholds:
            # Apply threshold to specific label
            adjusted_outputs = outputs[:, label_idx] >= threshold
            metric = metric_func(targets[:, label_idx], adjusted_outputs)
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
                
        optimal_thresholds.append(best_threshold)
    
    return optimal_thresholds

def freeze_layers(model, n: int = 1):
    """
    Helper function to freeze everything but the last num_layers of a transformer model.

    Assumes that the first layer, named l1, is the transformer model that the freeze operation is applied to.     
    """
    # Freeze all the parameters in the model first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last n layers of the BERT model
    for i, layer in enumerate(model.l1.encoder.layer[-n:], 1):
        for name, param in layer.named_parameters():
            print(f"Unfreezing {name}")
            param.requires_grad = True

    # Unfreeze all the parameters in the dropout and the last linear layers
    for param in model.l2.parameters():
        param.requires_grad = True
    for param in model.l3.parameters():
        param.requires_grad = True

    return model

def eval_metrics(targets, outputs, thresholds: list):
    # Ensure targets and outputs are numpy arrays for element-wise operations
    targets = np.array(targets)
    outputs = np.array(outputs)

    # Apply per-label thresholding
    # Assuming outputs and thresholds are appropriately aligned
    for i, threshold in enumerate(thresholds):
        outputs[:, i] = outputs[:, i] >= threshold
    
    # Calculate metrics after thresholding
    accuracy = metrics.accuracy_score(targets, outputs)
    hamming_loss = metrics.hamming_loss(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro', zero_division=np.nan)
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro', zero_division=np.nan)
    clf_report = metrics.classification_report(targets, outputs, zero_division=np.nan)

    # Print overall metrics
    print(f"Validation Accuracy Score = {accuracy}")
    print(f"Validation Hamming Loss = {hamming_loss}")
    print(f"Validation F1 Score (Micro) = {f1_score_micro}")
    print(f"Validation F1 Score (Macro) = {f1_score_macro}")
    print(clf_report)

    # Debugging: Write comparison of targets and outputs to file
    with open("debug-log.txt", "w") as log_file:
        log_file.write("Index, Target, Output, Subset Accuracy, Real Accuracy (%)\n")
        for index, (target_row, output_row) in enumerate(zip(targets, outputs)):
            # Here, the comparison is row-wise (per example, not per label)
            correct = np.array_equal(target_row, output_row)
            
            # Calculate subset accuracy
            subset_accuracy = (target_row == output_row).sum() / len(target_row)
            is_correct = 1 if correct else 0
            # Convert arrays to strings for logging
            target_str = np.array2string(target_row, separator=',')
            output_str = np.array2string(output_row, separator=',')
            log_file.write(f"{index}, {target_str}, {output_str}, {is_correct}, {subset_accuracy:.2f}%\n")