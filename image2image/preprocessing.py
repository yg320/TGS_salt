import glob
import numpy as np
import imageio
import os
import random
# from skimage.transform import resize
from scipy.misc import imresize

if __name__ == '__main__':
    random.seed(320)

    data_path = '/Users/yakirgorski/Documents/projects/TGS_salt/Data/all/train/'
    out_path = '/Users/yakirgorski/Documents/projects/TGS_salt/Data/processed/'

    set_names = ['train', 'validation']
    files = [os.path.basename(f) for f in glob.glob('/Users/yakirgorski/Documents/projects/TGS_salt/Data/all/train/images/*')]
    train_size = int(0.8 * len(files))
    random.shuffle(files)
    files = [files[: train_size], files[train_size:]]


    for set_name, set_files in zip(set_names, files):

        os.makedirs(os.path.join(out_path, set_name))

        for file in set_files:
            image_file = os.path.join(data_path, 'images', file)
            mask_file = os.path.join(data_path, 'masks', file)

            image = imageio.imread(image_file)
            mask = imageio.imread(mask_file)

            # TODO: verify why image has 3 channels
            assert set(np.unique(mask)) <= set([0, 65535])
            assert np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 0] == image[:, :, 2])

            mask[mask == 65535] = 255
            image = image[:, :, 0]

            # TODO: depracated
            image = imresize(image, (128, 128), interp='bilinear')
            mask = imresize(mask, (128,128), interp='nearest')

            out = np.concatenate([image, mask], axis=1).astype(np.uint8)

            imageio.imsave(os.path.join(out_path, set_name, file), out)
