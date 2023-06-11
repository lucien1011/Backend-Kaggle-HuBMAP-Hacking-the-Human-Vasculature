import segmentation_models_pytorch as smp

base_dir = 'data/'

seed = 42

classes = ['blood_vessel','glomerulus',]
activation = 'sigmoid'
device = 'cpu'

encoder = 'timm-efficientnet-b7'
encoder_weights = 'imagenet'

model_class = smp.Unet
model_args = dict(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=len(classes),
    activation=activation,
)

