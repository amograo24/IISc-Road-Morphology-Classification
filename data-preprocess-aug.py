import albumentations as A
import cv2

def resize_image(image_shp, target_size=512, train = False):
    """
    Resize the image to the target size
    :param image: The image to resize
    :param target_size: The target size
    :return: The resized image
    """
    h, w, _ = image_shp

    max_size = max(h, w)

    transform = A.Compose([
    A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.Resize(512, 512, interpolation=cv2.INTER_AREA)] + [A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.8),
        A.Rotate(limit=15, p=0.7),
    ],p=0.8)] if train else [])

    return transform

# %%
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)