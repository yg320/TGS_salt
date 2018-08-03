import numpy as np
import glob

def get_image_classification_type(IoU_threshold, mask_real, mask_generated):

    assert set(np.unique(mask_real)) <= {0, 1}
    assert set(np.unique(mask_generated)) <= {0, 1}

    if np.sum(mask_real) == 0:
        if np.sum(mask_generated) > 0:
            return np.array([0, 0, 0, 1])  # False Positive
        else:
            return np.array([0, 0, 1, 0])  # True Negative

    else:

        intersection = np.sum(mask_real + mask_generated == 2)
        union = np.sum(mask_real + mask_generated > 0)
        IoU = intersection / union

        assert IoU < 1.0
        if IoU < IoU_threshold:
            return np.array([0, 1, 0, 0])  # False Negative
        else:
            return np.array([1, 0, 0, 0])  # True Positive

def get_score_per_threshold(IoU_threshold):
    classification = np.array([0,0,0,0])

    for f in glob.glob('/Users/yakirgorski/Documents/projects/TGS_salt/Experiments/sample/*'):

        image = np.load(f)

        mask_real = image[:, 128:256] > 0
        mask_generates = image[:, 256:] > 220

        classification += get_image_classification_type(IoU_threshold, mask_real.astype(np.uint8), mask_generates.astype(np.uint8))

    return classification[0] / (classification[0] + classification[3] + classification[1])

score = 0.0

for IoU_threshold in np.arange(0.5,1,0.05):
    score += get_score_per_threshold(IoU_threshold)
    print(score / 10)