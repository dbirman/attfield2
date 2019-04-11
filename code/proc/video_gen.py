import numpy as np
import skvideo.io
from scipy import ndimage
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


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



def rf_slider(framesize, speed = 1., widths = [20, 5, 2]):
    videos = []
    for width_ratio in widths:
        width = framesize//width_ratio
        frames = int(8*framesize/np.sqrt(width)/speed)
        videos.append(sliding_bar_sequence(framesize, width, frames))
    return np.concatenate(videos)


def rf_slider_check(framesize, speed = 1, bar_widths = [20, 5, 2],
                      check_widths = [50, 15, 5]):
    videos = []
    slider = rf_slider(framesize, speed, widths = bar_widths)

    for check_ratio in check_widths:
        check_width = framesize//check_ratio
        for check_fn in [checker_sqare, checker_diamond]:
            check = check_fn(framesize, check_width)[np.newaxis, ...]
            videos.append(slider * (check * 255 - 127) + 127)
            videos.append(slider * (128 - check * 255) + 127)
    return np.concatenate(videos)



if __name__ == '__main__':
    # video = sliding_bar_sequence(100, 7, 23*8)
    

    # checker = checker_diamonond(246, 20)
    # plt.imshow(checker)
    # plt.colorbar()
    # plt.show()
    # exit()

    video = rf_slider_check(246)
    skvideo.io.vwrite("outputvideo.mp4", video)

