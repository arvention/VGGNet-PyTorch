import cv2
import torch
import numpy as np
from numpy import random


def intersect(box_a,
              box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    intersection = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

    return intersection[:, 0] * intersection[: 1]


def jaccard_numpy(box_a,
                  box_b):

    intersection = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union = area_a + area_b - intersection

    return intersection / union


class Compose(object):
    """This class applies a list of transformation to an image."""

    def __init__(self,
                 transforms):
        """Class constructor of Compose

        Arguments:
            transforms {list} -- list of transformation to be applied to the
            image
        """

        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- transformed image pixels, corresponding class
            of the image
        """

        for transform in self.transforms:
            image, labels = transform(image, labels)

        return image, labels


class ConvertToFloat(object):
    """This class casts an np.ndarray to floating-point data type."""

    def __init__(self):
        """Class constructor for ConvertToFloat"""

        super(ConvertToFloat, self).__init__()

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels casted to floating-point data type,
            corresponding class of the image
        """

        return image.astype(np.float32), labels


class SubtractMeans(object):
    """This class subtracts the mean from the pixel values of the image"""

    def __init__(self,
                 mean):
        """Class constructor for SubtractMeans

        Arguments:
            mean {tuple} -- mean
        """

        super(SubtractMeans, self).__init__()
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels subtracted by the mean,
            corresponding class of the image
        """

        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), labels


class Resize(object):
    """This class resizes the image to output size"""

    def __init__(self,
                 size=300):
        """Class constructor for Resize

        Keyword Arguments:
            size {int} -- output size (default: {300})
        """

        super(Resize, self).__init__()
        self.size = size

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- pixels of the resized image, corresponding class
            of the image
        """

        image = cv2.resize(image, (self.size, self.size))
        return image, labels


class RandomSaturation(object):
    """This class adjusts the saturation of the image. This expects an image
    in HSV color space, where the 2nd channel (saturation channel) should have
    values between 0.0 to 1.0"""

    def __init__(self,
                 lower=0.5,
                 upper=1.5):
        """Class constructor for RandomSaturation

        Keyword Arguments:
            lower {int} -- lower bound of the interval used in generating
            a random number from a uniform distribution to adjust saturation
            (default: {0.5})
            upper {number} -- upper bound of the interval used in generating
            a random number from a uniform distribution to adjust saturation.
            (default: {1.5})
        """

        super(RandomSaturation, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray. This
            should be in HSV color space.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels in HSV color space with adjusted
            saturation channel, corresponding class of the image
        """

        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

            # limits the value of the saturation channel to 1.0
            image[:, :, 1] = np.clip(image[:, :, 1], a_min=0.0, a_max=1.0)

        return image, labels


class RandomHue(object):
    """This class adjusts the hue of the image. This expects an image
    in HSV color space, where the 1st channel (hue channel) should have
    values between 0.0 to 360.0"""

    def __init__(self,
                 delta=18.0):
        """Class constructor for RandomSaturation

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            hue (default: {18.0})
        """

        super(RandomHue, self).__init__()
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray. This
            should be in HSV color space.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels in HSV color space with adjusted
            hue channel, corresponding class of the image
        """

        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)

            # limits the value of the hue channel to 360
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, labels


class RandomLightingNoise(object):
    """This class randomly swaps the channels of the image to create a
    lighting noise effect. This class calls the class SwapChannels."""

    def __init__(self):
        """Class constructor for RandomLightingNoise"""

        super(RandomLightingNoise, self).__init__()
        self.permutations = ((0, 1, 2), (0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- pixels of the image with swapped channels,
            corresponding class of the image
        """

        if random.randint(2):
            swap = self.permutations[random.randint(len(self.permutations))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)

        return image, labels


class ConvertColor(object):
    """This class converts the image to another color space. This class
    supports the conversion from BGT to HSV color space and vice-versa."""

    def __init__(self,
                 current='BGR',
                 transform='HSV'):
        """Class constructor for ConvertColor

        Keyword Arguments:
            current {str} -- the input color space of the image
            (default: {'BGR'})
            transform {str} -- the output color space of the image
            (default: {'HSV'})
        """

        super(ConvertColor, self).__init__()
        self.current = current
        self.transform = transform

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- pixels of the image converted to the output
            color space, corresponding class of the image
        """

        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        return image, labels


class RandomContrast(object):
    """This class adjusts the contrast of the image. This multiplies a random
    constant to the pixel values of the image."""

    def __init__(self,
                 lower=0.5,
                 upper=1.5):
        """Class constructor for RandomContrast

        Keyword Arguments:
            lower {int} -- lower bound of the interval used in generating
            a random number from a uniform distribution to adjust contrast
            (default: {0.5})
            upper {number} -- upper bound of the interval used in generating
            a random number from a uniform distribution to adjust contrast
            (default: {1.5})
        """

        super(RandomContrast, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels with adjusted contrast,
            corresponding class of the image
        """

        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)

            # multiplies the random constant to the pixel values
            image *= alpha

        return image, labels


class RandomBrightness(object):
    """This class adjusts the brightness of the image. This adds a random
    constant to the pixel values of the image."""

    def __init__(self,
                 delta=32.0):
        """Class constructor for RandomBrightness

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            brightness (default: {32.0})
        """

        super(RandomBrightness, self).__init__()
        assert delta >= 0.0 and delta <= 255.0
        self.delta = delta

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels with adjusted brightness,
            corresponding class of the image
        """

        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)

            # adds the random constant to the pixel values
            image += delta

        return image, labels


class ToCV2Image(object):
    """This class converts the torch.Tensor representation of an image to
    np.ndarray. The channels of the image are also converted from RGB to
    BGR"""

    def __init__(self):
        """Class constructor for ToCV2Image"""

        super(ToCV2Image, self).__init__()

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {torch.Tensor} -- image pixels represented as torch.Tensor.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels represented as np.ndarray,
            corresponding class of the image
        """

        # permute() is used to switch the channels from RGB to BGR
        image = image.cpu().numpy().astype(np.float32).transpose((2, 1, 0))
        return image, labels


class ToTensor(object):
    """This class converts the np.ndarray representation of an image to
    torch.Tensor. The channels of the image are also converted from BGR to
    RGB"""

    def __init__(self):
        """Class constructor for ToTensor"""

        super(ToTensor, self).__init__()

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            torch.Tensor, int -- image pixels represented as torch.Tensor,
            corresponding class of the image
        """

        # permute() is used to switch the channels from BGR to RGB
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 1, 0)
        return image, labels


class RandomSampleCrop(object):
    """This class randomly crops the image. This gets a random crop based on a
    minimum scale."""

    def __init__(self,
                 min_scale=0.7):
        """Class constructor for RandomSampleCrop

        Keyword Arguments:
            min_scale {int} -- Minimum scale to crop from the image
            (default: {0.7})
        """

        super(RandomSampleCrop, self).__init__()
        self.min_scale = min_scale

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {torch.Tensor} -- image pixels represented as torch.Tensor.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels of the randomly cropped image,
            corresponding class of the image
        """

        if random.randint(2):
            h, w, _ = image.shape

            scale = np.random.uniform(low=self.min_scale, high=1.0)
            size = int(scale * np.amin([h, w]))

            top_x_range = [0, w - size]
            top_y_range = [0, h - size]
            top_x = np.random.randint(top_x_range[0], top_x_range[1])
            top_y = np.random.randint(top_y_range[0], top_y_range[1])

            image = image[top_y:top_y + size, top_x:top_x + size, :]

        return image, labels


class RandomMirror(object):
    """This class randomly flips the image horizontally."""

    def __init__(self):
        """Class constructor for RandomMirror"""

        super(RandomMirror, self).__init__()

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels flipped horizontally, corresponding
            class of the image
        """

        _, width, _ = image.shape

        if random.randint(2):
            image = image[:, ::-1, :]

        return image, labels


class SwapChannels(object):
    """This class swaps the channels of the image"""

    def __init__(self,
                 swaps):
        """Class constructor for SwapChannels

        Arguments:
            swaps {tuple} -- new order of the channels
        """

        super(SwapChannels, self).__init__()
        self.swaps = swaps

    def __call__(self,
                 image):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Returns:
            np.ndarray -- pixels of the image with swapped channels
        """

        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """This class applies different transformation to the image. This includes
    adjustment of contrast, saturation, hue, and brightness, and the
    switching of channels."""

    def __init__(self):
        """Class constructor for PhotometricDistort"""

        super(PhotometricDistort, self).__init__()
        self.pd = [RandomContrast(),
                   ConvertColor(transform='HSV'),
                   RandomSaturation(),
                   RandomHue(),
                   ConvertColor(current='HSV', transform='BGR'),
                   RandomContrast()]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self,
                 image,
                 labels):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image

        Returns:
            np.ndarray, int -- image pixels applied with different random
            transformations, corresponding class of the image
        """

        image = image.copy()
        image, labels = self.rand_brightness(image, labels)

        # applies RandomContrast() as the first transformation
        if random.randint(2):
            distort = Compose(self.pd[:-1])

        # applies RandomContrast() as the last transformation
        else:
            distort = Compose(self.pd[1:])
        image, labels = distort(image, labels)

        return self.rand_light_noise(image, labels)


class Augmentations(object):
    """This class applies different augmentation techniques to the image.
    This is used for training the model."""

    def __init__(self,
                 size,
                 mean):
        """Class constructor for Augmentations

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(Augmentations, self).__init__()
        self.size = size
        self.mean = mean
        self.augment = Compose([ConvertToFloat(),
                                PhotometricDistort(),
                                RandomSampleCrop(),
                                RandomMirror(),
                                Resize(self.size),
                                SubtractMeans(self.mean)])

    def __call__(self,
                 image,
                 labels):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image

        Returns:
            np.ndarray, int -- image pixels applied with different augmentation
            techniques, corresponding class of the image
        """

        return self.augment(image, labels)


class BaseTransform(object):
    """This class applies different base transformation techniques to the
    image. This includes resizing the image and subtracting the mean from the
    image pixels. This is used for testing the model."""

    def __init__(self,
                 size,
                 mean):
        """Class constructor for BaseTransform

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(BaseTransform, self).__init__()
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            labels {np.ndarray} -- corresponding class of the image
            (default: {None})

        Returns:
            np.ndarray, int -- image pixels applied with different augmentation
            techniques, corresponding class of the image
        """

        dimensions = (self.size, self.size)
        image = cv2.resize(image, dimensions).astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32)
        return image, labels
