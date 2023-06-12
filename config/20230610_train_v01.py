import segmentation_models_pytorch as smp

# IO
base_dir = 'data/'
tag = '20230610_train_v01'
version = 'v02'

# Misc
seed = 42
device = 'cpu'

# Data
classes = ['blood_vessel','glomerulus',]

# Model
activation = 'sigmoid'
encoder = 'timm-efficientnet-b0'
encoder_weights = 'noisy-student'
model_class = smp.Unet
model_args = dict(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=len(classes),
    activation=activation,
)

pr_threshold = 0.6
