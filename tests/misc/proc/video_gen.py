from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import scipy.ndimage.interpolation as scimg
import numpy as np
import skvideo.io
from scipy import ndimage
from enum import Enum
from torchvision import transforms
import torch
import cv2
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

def batch_transform(batch, transform):
    newbatch = [transform(batch[i]) for i in range(len(batch))]
    return torch.stack(newbatch)

def noise_video(frames, channels, width, height, numpy = True):
    inputs = np.random.randint(0, 255,
        size = (frames, channels, width, height))
    inputs = torch.from_numpy(inputs).float()
    inputs = batch_transform(inputs, normalize)
    if numpy: inputs = inputs.numpy()
    return inputs


def sliding_bar_lr(framesize, width, frames):
    positions = np.linspace(-width, framesize, frames)
    X = np.arange(framesize)
    video = np.apply_along_axis(
        lambda pos: (pos <= X) & (X < pos+width),
        0, positions.reshape([1, -1])
        )
    video = np.tile(video, [framesize, 1, 1])
    return video.transpose([2, 0, 1])

def sliding_bar_diag(framesize, width, frames):
    width = width*np.sqrt(2)
    positions = np.linspace(-width, framesize*2, frames)
    X, Y = np.meshgrid(np.arange(framesize), np.arange(framesize))
    X = X + Y
    video = np.apply_along_axis(
        lambda pos: (pos <= X) & (X < pos+width),
        0, positions.reshape([1, -1])
        )
    return video.transpose([2, 0, 1])


class CardinalDirection(Enum):
    LEFT = 0
    TOPLEFT = 1
    TOP = 2
    TOPRIGHT = 3
    RIGHT = 4
    BOTTOMRIGHT = 5
    BOTTOM = 6
    BOTTOMLEFT = 7
CardinalDirection.ORTHO = [
    CardinalDirection.LEFT,
    CardinalDirection.TOP,
    CardinalDirection.RIGHT,
    CardinalDirection.BOTTOM
]
CardinalDirection.DIAG = [
    CardinalDirection.TOPLEFT,
    CardinalDirection.TOPRIGHT,
    CardinalDirection.BOTTOMRIGHT,
    CardinalDirection.BOTTOMLEFT
]



def sliding_bar(framesize, width, frames, angle):
    '''
    Produce a square black frame with a white bar sliding across

    ### Arguments
    - `framesize` --- Integer giving width/height of frame
    - `width` --- Width of the bar
    - `frames` --- Number of frames it should take the bar to move
        across the video frame.
    - `angle` --- A CardinalDirection, where the bar should start.

    ### Returns
    - `video` --- Binary numpy array with shape (frames, framesize,
        framesize)

    ### Raises
    - `ValueError` --- If an invalid CardinalDirection was given.
    '''

    if angle in CardinalDirection.ORTHO:
        video = sliding_bar_lr(framesize, width, frames)
    elif angle in CardinalDirection.DIAG:
        video = sliding_bar_diag(framesize, width, frames)
    else:
        raise ValueError("Invalid CardinalDirection:" + repr(angle))

    if angle in [CardinalDirection.RIGHT, CardinalDirection.TOPRIGHT]:
        return video[:, :, ::-1]
    elif angle in [CardinalDirection.TOP]:
        return video.transpose([0, 2, 1])
    elif angle in [CardinalDirection.BOTTOM]:
        return video.transpose([0, 2, 1])[:, ::-1, :]
    elif angle in [CardinalDirection.BOTTOMRIGHT]:
        return video[:, ::-1, ::-1]
    elif angle in [CardinalDirection.BOTTOMLEFT]:
        return video[:, ::-1, :]
    else:
        return video


def sliding_bar_sequence(framesize, width, frames):
    '''
    Produce a square black frame with a white bar sliding across in
    each of the eight cardinal directions

    ### Arguments
    - `framesize` --- Integer giving width/height of frame
    - `width` --- Width of the bar
    - `frames` --- Number of frames it should take the bar to move
        across the video frame eight times.

    ### Returns
    - `video` --- Binary numpy array with shape (frames, framesize,
        framesize)
    '''
    videos = [sliding_bar(framesize, width, frames//8, angle)
              for i, angle in enumerate(CardinalDirection)]
    return np.concatenate(videos)


def checker_sqare(framesize, width):
    mask = (np.arange(framesize) % (width*2)) < width
    X, Y = np.meshgrid(mask, mask.copy())
    return np.int8(X ^ Y)

def checker_diamond(framesize, width):
    width = int(np.sqrt(2) * width)
    X, Y = np.meshgrid(np.arange(framesize), np.arange(framesize))
    dim1 = ((X+Y) % (width*2)) < width
    dim2 = ((X-Y) % (width*2)) < width
    return np.int8(dim1 ^ dim2)



def rf_slider(framesize, speed = 1., widths = [20, 5, 2], channels = 3):
    videos = []
    for width_ratio in widths:
        width = framesize//width_ratio
        frames = int(8*framesize/np.sqrt(width)/speed)
        videos.append(sliding_bar_sequence(framesize, width, frames))
    return np.concatenate(videos)


def rf_slider_check(framesize, speed = 1, bar_widths = [20, 5, 2],
                      check_widths = [50, 15, 5], channels = 3):
    videos = []
    slider = rf_slider(framesize, speed, widths = bar_widths)

    for check_ratio in check_widths:
        check_width = framesize//check_ratio
        for check_fn in [checker_sqare, checker_diamond]:
            check = check_fn(framesize, check_width)[np.newaxis, ...]
            videos.append(slider * (check * 255 - 127) + 127)
            videos.append(slider * (128 - check * 255) + 127)
    video = np.concatenate(videos)
    video = np.tile(video[:, np.newaxis, :, :], [1, channels, 1, 1])
    return batch_transform(torch.tensor(video).float(), normalize)


def cifar_images(framesize, n):
    '''Center and rotate cifar10 images for voxels'''
    if not hasattr(cifar_images, 'cifar'):
        from keras.datasets import cifar10
        (x_train, y_train), _ = cifar10.load_data()
        cifar_images.cifar = x_train

    frames = []
    for i in range(n):
        idx = np.random.randint(len(x_train))
        n_tile = 2*framesize//32+1
        double_frame = np.tile(x_train[idx], [n_tile, n_tile, 1])

        theta = np.random.uniform(low = -180, high=180)
        t_x = np.random.uniform(low = -16, high = 16)
        t_y = np.random.uniform(low = -16, high = 16)
        M = cv2.getRotationMatrix2D((framesize+t_x,framesize+t_y),theta,1)
        double_frame = cv2.warpAffine(double_frame, M, (framesize*2, framesize*2))
        frame = double_frame[framesize//2-16:framesize+framesize//2+16,
                             framesize//2-16:framesize+framesize//2+16, :]
        frames.append(frame)
    video = np.moveaxis(np.array(frames), -1, 1)
    return batch_transform(torch.tensor(video).float(), normalize)

def uncache_cifar():
    del cifar_images.cifar



def lab_colors(thetas, L = 50):
    As = np.cos(thetas)*100
    Bs = np.sin(thetas)*100
    colors = [convert_color(LabColor(L, a, b), sRGBColor)
              for a, b in zip(As, Bs)]
    colors = np.array([
        (c.clamped_rgb_r, c.clamped_rgb_g, c.clamped_rgb_b)
        for c in colors])
    return colors


def color_rotation(framesize, frames):
    thetas = np.linspace(0, 2*np.pi, frames + 1)[:-1]
    colors = lab_colors(thetas)
    base = np.ones([frames, framesize, framesize, 3])
    colors_shaped = colors[:, np.newaxis, np.newaxis, :]
    video = base * colors_shaped * 255
    skvideo.io.vwrite('data/color_output.mp4', video)
    video = np.moveaxis(video, -1, 1)
    return batch_transform(torch.tensor(video).float(), normalize)


def sine_rotation(framesize, frames, freqs = [20]):
    thetas = np.linspace(0, np.pi, frames)
    Bs = np.cos(thetas)[:, np.newaxis, np.newaxis]
    As = np.sin(thetas)[:, np.newaxis, np.newaxis]
    basic_x = np.arange(framesize)/framesize-0.5
    basic_y = basic_x.copy()
    X, Y = np.meshgrid(basic_x, basic_y)
    grid = As * X[np.newaxis, ...] + Bs * Y[np.newaxis, ...]

    videos = [np.sin((2*np.pi*F) * grid)/2 + 0.5
              for F in freqs]
    video = np.tile(np.concatenate(videos)[:, np.newaxis, :, :],
                    [1, 3, 1, 1]) * 255
    return batch_transform(torch.tensor(video).float(), normalize)




MCGILL_CLASSES = ['Flowers', 'Fruits', 'LandWater',
                  'Textures', 'Foliage', 'ManMade']

def mcgill_images(framesize, n, db = 'data/mcgill',
    classes = MCGILL_CLASSES):
    '''
    ### Arguments
    - `n` --- Number of images per category
    '''
    frames = []
    for c in classes:
        imgs = [s for s in os.listdir(os.path.join(db, c))
                if s.endswith('.tif')]
        select = np.random.choice(len(imgs), n, replace = False)
        for i in select:
            # Find the image path and read
            path = os.path.join(db, c, imgs[i])
            frame = skvideo.io.vread(path)[0]
            # Zoom to framesize
            factor = framesize/min(frame.shape[0], frame.shape[1])
            factor *= np.random.uniform(low = 1.1, high = 1.33)
            frame = scimg.zoom(frame, [factor, factor, 1])
            # Random crop to exact framesize
            x_crop = np.random.randint(
                low = 0, high = frame.shape[0] - framesize)
            y_crop = np.random.randint(
                low = 0, high = frame.shape[1] - framesize)
            frame = frame[x_crop:x_crop + framesize,
                          y_crop:y_crop + framesize, :]
            frames.append(frame)
    video = np.moveaxis(frames, -1, 1)
    return batch_transform(torch.tensor(video).float(), normalize)


def mcgill_groups(n, classes = MCGILL_CLASSES):
    groups = np.array(classes)
    return np.repeat(groups, n)


if __name__ == '__main__':
    # video = sliding_bar_sequence(100, 7, 23*8)
    
    # checker = checker_diamonond(246, 20)
    # plt.imshow(checker)
    # plt.colorbar()
    # plt.show()
    # exit()

    #video = mcgill_images(64, 5)
    skvideo.io.vwrite("data/test_video.mp4", video)













