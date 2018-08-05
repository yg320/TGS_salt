import glob
import numpy as np
import imageio
import os
import random
# from skimage.transform import resize
from scipy.misc import imresize
import csv

if __name__ == '__main__':
    random.seed(320)

<<<<<<< HEAD
    data_path = '/Users/yakirgorski/Documents/projects/TGS_salt/Data/all/train/'
    out_path = '/Users/yakirgorski/Documents/projects/TGS_salt/Data/processed_aug_with_depth/'
=======
    data_path = '/home/paperspace/Data/TGS_salt/all/train/'
    out_path = '/home/paperspace/Data/TGS_salt/processed_aug/'
>>>>>>> 106861e141ee543234207d891327d06dc5b004a1

    set_names = ['train', 'validation']
    files = [os.path.basename(f) for f in glob.glob(os.path.join(data_path, 'images/*'))]
    train_size = int(0.8 * len(files))
    random.shuffle(files)
    files = [files[: train_size], files[train_size:]]

    with open('/Users/yakirgorski/Documents/projects/TGS_salt/Data/all/depths.csv', 'r') as f:
        reader = csv.reader(f)
        depths = list(reader)
        depth_min = float(np.max([int(x[1]) for x in depths[1:]]))
        depth_max = float(np.min([int(x[1]) for x in depths[1:]]))
        depths = dict(depths)


    def no_op(a):
        return a

    for set_name, set_files in zip(set_names, files):

        os.makedirs(os.path.join(out_path, set_name))

        for file in set_files:
            image_file = os.path.join(data_path, 'images', file)
            mask_file = os.path.join(data_path, 'masks', file)
            depth = float(depths[os.path.splitext(file)[0]])
            normalized_depth = np.uint8(255.0 * (depth - depth_min) / (depth_max - depth_min))

            image = imageio.imread(image_file)
            mask = imageio.imread(mask_file)

            # TODO: verify why image has 3 channels
            assert set(np.unique(mask)) <= set([0, 65535])
            assert np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 0] == image[:, :, 2])

            mask[mask == 65535] = 1
            image = image[:, :, 0]

            # TODO: depracated
            image = imresize(image, (160, 160), interp='bilinear')
            mask = imresize(mask, (160,160), interp='nearest')
            depth_im = normalized_depth * np.ones_like(mask)

            if set_name == 'train':
                counter = 0
                for fliplr in [np.fliplr, no_op]:
                    for flipud in [np.flipud, no_op]:
                        for rot90 in [np.rot90, no_op]:

                            out = np.concatenate([fliplr(flipud(rot90(image))), fliplr(flipud(rot90(mask))), depth_im], axis=1).astype(np.uint8)

                            imageio.imsave(os.path.join(out_path, set_name, f'{counter}_{file}'), out)
                            counter += 1
            else:
                out = np.concatenate([image, mask, depth_im], axis=1).astype(np.uint8)

                imageio.imsave(os.path.join(out_path, set_name, file), out)