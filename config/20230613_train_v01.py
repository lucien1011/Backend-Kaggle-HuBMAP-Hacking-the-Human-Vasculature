import segmentation_models_pytorch as smp

# IO
base_dir = 'data/'
tag = '20230613_train_v01'
#version = 'Unet_timm-efficientnet-b0_noisy-student_sigmoid_cls1_5e-05_AdamW_None'
#version = 'Unet_timm-efficientnet-b0_noisy-student_sigmoid_cls1_5e-05_AdamW_None_8aug'
version = 'Unet_timm-efficientnet-b7_noisy-student_sigmoid_cls1_5e-05_AdamW_None_spatialaug5'

# Misc
seed = 42
device = 'cpu'

# Data
classes = ['blood_vessel']

# Model
activation = 'sigmoid'
encoder = 'timm-efficientnet-b7'
encoder_weights = 'noisy-student'
model_class = smp.Unet
model_args = dict(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=len(classes),
    activation=activation,
)

pr_threshold = 0.95
