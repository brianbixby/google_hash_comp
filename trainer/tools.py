import numpy as np


def rand_map(hight, width):
    lst = ["T", "M"]
    map = np.random.choice(lst, hight*width)
    # map = np.reshape(map, (hight, width))

    return map

# def