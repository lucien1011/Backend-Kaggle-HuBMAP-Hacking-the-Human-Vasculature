import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

class Displayer(object):
    CLASSES = ['blood_vessel', 'glomerulus', 'unsure']
    
    def __init__(
            self, 
            image_map, 
            annotations, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):

        self.image_map = image_map
        self.image_keys = list(self.image_map.keys())
        self.annotations = annotations

        self.classes = self.CLASSES if classes is None else classes
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def display_image(self,image_id,custom_masks=None,output_path=None):
        image = cv2.imread(self.image_map[image_id])
        overlay = image.copy()

        if image_id in self.annotations:
            polygons = self.annotations[image_id]

        assert polygons is not None

        masks = {cls:np.zeros((512,512)) for cls in self.classes}
        for polygon in polygons:
            cls = polygon['type']
            coordinates = polygon['coordinates']
            if cls not in self.classes: continue
            lines = np.array(polygon['coordinates'])
            lines = lines.reshape(-1, 1, 2)
            cv2.fillPoly(masks[cls], [lines], 1)

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

        true_overlay = image.copy()
        annotation_counts = {}
        for idx, (annotation_type, mask) in enumerate(masks.items(), 2):
            labeled, num_areas = label(mask)
            annotation_counts[annotation_type] = num_areas
            true_overlay[mask > 0] = colors[idx-2]

        nplots = 3 if custom_masks is not None else 2
        fig,ax = plt.subplots(1,nplots,figsize=(10*nplots,10))
        ax[0].imshow(image)
        ax[1].imshow(image)
        ax[1].imshow(true_overlay,alpha=0.4)
        if custom_masks is not None:
            custom_overlay = image.copy()
            annotation_counts = {}
            for idx, (annotation_type, mask) in enumerate(custom_masks.items(), 2):
                labeled, num_areas = label(mask)
                annotation_counts[annotation_type] = num_areas
                custom_overlay[mask > 0] = colors[idx-2]
            ax[2].imshow(image)
            ax[2].imshow(custom_overlay,alpha=0.4)

        fig.tight_layout()

        if output_path: fig.savefig(output_path)
        plt.close(fig)

