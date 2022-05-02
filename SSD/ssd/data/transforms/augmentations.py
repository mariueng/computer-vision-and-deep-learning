import cv2
import torch
import numpy as np
from numpy import random


"""

Implementation of methods and photometric distortion originally from https://github.com/lufficc/SSD

Most methods are rewritten with a torch wrapper to match the samples in the dataset (image, boxes, labels)

"""

class SwapChannels(torch.nn.Module):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        super().__init__()
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        if torch.is_tensor(image):
            image = image.data.cpu().numpy()
        image = image[:, :, self.swaps]
        # TODO: Needs to output in the following format:
        # [1, 3, 128, 1024] not torch.Size([1, 128, 1024, 3])
        return image

    
# The next two methods are applied outside the distort loop.
class RandomBrightness(torch.nn.Module):
    """Adjust brightness of an image.
    Args:
        sample: 
    """
    def __init__(self, delta=32):
        super().__init__()
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        image = sample["image"]
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        sample["image"] = image
        return sample

    
class RandomLightingNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, sample):
        image = sample["image"]
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        sample["image"] = image
        return sample


# The next methods are applied in the loop
class RandomSaturation(torch.nn.Module):
    def __init__(self, lower=0.5, upper=1.5):
        super().__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, sample):
        image = sample["image"]
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        sample["image"] = image
        return sample


class RandomHue(torch.nn.Module):
    def __init__(self, delta=18.0):
        super().__init__()
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, sample):
        image = sample["image"]
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        sample["image"] = image
        return sample


class RandomContrast(torch.nn.Module):
    def __init__(self, lower=0.5, upper=1.5):
        super().__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        image = sample["image"]
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        sample["image"] = image
        return sample


class ConvertColor(torch.nn.Module):
    def __init__(self, current, transform):
        super().__init__()
        self.transform = transform
        self.current = current

    def __call__(self, sample):
        image = sample["image"]

        # Check if image is tensor, in case transform to ndarray
        if torch.is_tensor(image):
            image = image.cpu().numpy()

        # Transpose ndarray shape for cv2 converter
        image = image.astype(np.float32).transpose((1, 2, 0))
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError

        # Transpose ndarray back to original shape
        sample["image"] = image.transpose((2, 0, 1))
        return sample


# def remove_empty_boxes(boxes, labels):
#     """Removes bounding boxes of W or H equal to 0 and its labels
#     Args:
#         boxes   (ndarray): NP Array with bounding boxes as lines
#                            * BBOX[x1, y1, x2, y2]
#         labels  (labels): Corresponding labels with boxes
#     Returns:
#         ndarray: Valid bounding boxes
#         ndarray: Corresponding labels
#     """
#     del_boxes = []
#     for idx, box in enumerate(boxes):
#         if box[0] == box[2] or box[1] == box[3]:
#             del_boxes.append(idx)

#     return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)


class Compose(torch.nn.Module):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class PhotometricDistort(torch.nn.Module):
    """Photometric distortion. Randomly change the brightness, contrast, saturation and hue of an image.
    Args:
        sample:
    returns:
        sample:
    """
    def __init__(self):
        super().__init__()
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, sample):
        smp = sample.copy()
        # smp = self.rand_brightness(smp)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        smp = distort(smp)
        # smp = self.rand_light_noise(smp)
        return smp
