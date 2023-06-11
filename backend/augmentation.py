import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.Rotate(p=0.5),
        albu.GaussNoise(p=0.25),
        albu.Sharpen(p=0.25),
        albu.ColorJitter(p=0.25),
        #albu.Downscale(p=0.25),
        #albu.Emboss(p=0.25),
        #albu.GaussNoise(p=0.25),
        #albu.HueSaturationValue(p=0.25),
        #albu.MultiplicativeNoise(p=0.25),
        #albu.Normalize(p=1.0),
        #albu.PixelDropout(p=0.5),
        #albu.RandomBrightness(p=0.25),
        #albu.RandomBrightnessContrast(p=0.25),
        #albu.RandomContrast(p=0.25),
        #albu.RandomFog(p=0.25),
        #albu.ToGray(p=0.25),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
