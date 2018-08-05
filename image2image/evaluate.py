import numpy as np
import glob

TRUE_POSITIVE = 0
TRUE_NEGATIVE = 1
FALSE_POSITIVE = 2
FALSE_NEGATIVE = 3

def get_image_classification_type(IoU_threshold, mask_real, mask_generated):

    assert set(np.unique(mask_real)) <= {0, 1}
    assert set(np.unique(mask_generated)) <= {0, 1}

    if np.sum(mask_real) == 0:
        if np.sum(mask_generated) > 0:
            return FALSE_POSITIVE  # False Positive
        else:
            return TRUE_NEGATIVE  # True Negative

    else:

        intersection = np.sum(mask_real + mask_generated == 2)
        union = np.sum(mask_real + mask_generated > 0)
        IoU = intersection / union

        assert IoU <= 1.0
        if IoU < IoU_threshold:
            return FALSE_NEGATIVE  # False Negative
        else:
            return TRUE_POSITIVE  # True Positive

def get_score_per_threshold(IoU_threshold, softmax_threshold = 0.5):

    false_negative = 0
    false_positive = 0
    true_negative = 0
    true_positive = 0

    for f in glob.glob('/Users/yakirgorski/Documents/projects/TGS_salt/Experiments/4/sample/train/*'):
        image = np.load(f)

        mask_real = image[:, 128:256] > 0
        mask_generated = (1.0 - image[:, 256:]) > softmax_threshold

        classification_type = get_image_classification_type(IoU_threshold, mask_real.astype(np.uint8), mask_generated.astype(np.uint8))

        if classification_type == FALSE_NEGATIVE:
            false_negative += 1
        elif classification_type == FALSE_POSITIVE:
            false_positive += 1
        elif classification_type == TRUE_NEGATIVE:
            true_negative += 1
        elif classification_type == TRUE_POSITIVE:
            true_positive += 1

    return true_positive / (true_positive + false_positive + false_negative)

score = 0.0

for t in [0.5]:
    for IoU_threshold in np.arange(0.5,1,0.05):
        score += get_score_per_threshold(IoU_threshold, t)
        print(t, score / 10)