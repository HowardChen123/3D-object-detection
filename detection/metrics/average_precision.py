from dataclasses import dataclass
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.
    # EvaluationFrame: detections: Detections + labels: Detections
    # Detections: centroids: torch.Tensor + yaws: torch.Tensor + boxes: torch.Tensor + scores: Optional[torch.Tensor] = None
    # Detections: centroids_x(self) + centroids_y(self) + boxes_x(self) + boxes_y(self) + to(self, device: torch.device) + __len__(self)
    precision = []
    recall = []
    for evaluation in frames:
        detections = evaluation.detections.centroids
        labels = evaluation.labels.centroids
        TP = 0 
        FP = 0
        FN = 0
        cdist = torch.cdist(detections, labels, p=2)
        # print(torch.sum(cdist <= threshold)) # 216
        # record for labels matching
        labels_record = torch.zeros(labels.size()[0])
        for i in range(detections.size()[0]):
            # Computing TP/FP 
            # Step 1: the Euclidean distance between their centers is at most `threshold`; 
            # Get labels for one detection satisfying threshold
            under = cdist[i] <= threshold
            
            under_ind = under.nonzero() #torch.Size([n, 1])
            # Step 2: no higher scoring detection satisfies condition step1 with respect to the same label.
            matched_labels = cdist[i][under_ind] >= (cdist[:, under_ind] <= threshold)#torch.Size([1, 1]) torch.Size([500, 1, 1]) torch.Size([500, 1, 1])
            # print("before", matched_labels.size())
            # matched_labels = matched_labels.permute(1, 0, 2)
            # print("after", matched_labels.size())
            # Check if the detection satisfying any labels
            true_labels = torch.any(torch.all(matched_labels, 0))
            # print(1, torch.all(matched_labels, 0))
            # print(2, true_labels)
            # result = torch.any(torch.all(matched_labels, 1))
            result = torch.sum(true_labels)
            TP += result
            FP += 1 - result

            # Computing FN
            # record matched detections for each label
            labels_det = torch.all(matched_labels, 0)
            # Get index of all matched labels
            lables_ind = under_ind[labels_det] 
            # Update record of labels
            labels_record[lables_ind] += 1
        FN += labels.size()[0] - torch.count_nonzero(labels_record)
        precision.append(TP / (TP + FP))
        recall.append(TP / (TP + FN))
    print(1, len(precision), len(recall))
    return PRCurve(torch.tensor(precision), torch.tensor(recall))


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    # return torch.sum(curve.recall).item() * 0.0

    # PRCurve: precision: torch.Tensor + recall: torch.Tensor
    # Follow instructions, r_0 = 0.0
    # Prepare p
    p = curve.precision
    # Prepare r
    r = curve.recall
    # copy r
    c_range  = [i for i in range(0, r.size()[0] - 1)]
    range_tensor = torch.Tensor(c_range).int()
    copy = torch.zeros(r.size())
    c = copy.index_add_(0, range_tensor + 1, r[c_range])
    # r_i - r_{i - 1}
    r_minus = r - c
    AP = torch.sum(r_minus * p)
    return AP


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    PR = compute_precision_recall_curve(
        frames, threshold
    )  # return PRCurve(ap: float; pr_curve: PRCurve)
    AP = compute_area_under_curve(PR)  # return float AP
    return AveragePrecisionMetric(AP, PR)
