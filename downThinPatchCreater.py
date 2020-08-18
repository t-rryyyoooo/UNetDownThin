import SimpleITK as sitk
import cloudpickle
import torch
from functions import resampleSize, cropping, rounding, padding, caluculatePaddingSize, getImageWithMeta
import numpy as np
from tqdm import tqdm
from pathlib import Path

class DownThinPatchCreater():
    def __init__(self, image, label_patch_size, plane_size, overlap, num_down, num_rep=1, is_label=False, mask=None):
        self.image = image
        self.label_patch_size = np.array(label_patch_size)
        self.plane_size = np.array(plane_size)
        self.overlap = overlap
        self.num_down = num_down
        self.is_label = is_label
        self.num_rep = num_rep
        self.mask = mask

    def execute(self):
        # Crop or pad the image for required_shape.
        image = self.image
        mask = self.mask
        print("Image shape : {}".format(self.image.GetSize()))

        image_shape = np.array(self.image.GetSize())
        required_shape = np.array(self.image.GetSize())
        required_shape[0:2] = self.plane_size
        self.diff = required_shape - image_shape
        if (self.diff < 0).any():
            lower_crop_size = (abs(self.diff) // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in abs(self.diff) / 2]

            self.image = cropping(self.image, lower_crop_size, upper_crop_size)
    
            if mask is not None:
                mask = cropping(mask, lower_crop_size, upper_crop_size)

        else:
            lower_pad_size = (self.diff // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in self.diff / 2]

            self.image = padding(self.image, lower_pad_size, upper_pad_size)
            if mask is not None:
                mask = padding(mask, lower_pad_size, upper_pad_size)

        print("Image shape : {}".format(self.image.GetSize()))

        image_shape = np.array(self.image.GetSize())
        slide = self.label_patch_size // np.array((1, 1, self.overlap))
        self.axial_lower_pad_size, self.axial_upper_pad_size = caluculatePaddingSize(image_shape, self.label_patch_size, self.label_patch_size, slide)
        self.image = padding(self.image, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())
        if mask is not None:
            mask = padding(mask, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())
        print("Image shape : {}".format(self.image.GetSize()))

        # Downsample the image to one in num_down.
        required_shape = np.array(self.image.GetSize()) // 2**self.num_down
        self.image_shape = self.image.GetSize()
        if not self.is_label:
            self.image = resampleSize(self.image, required_shape.tolist(), is_label=False)
        else:
            self.image = resampleSize(self.image, required_shape.tolist(), is_label=True)
        if mask is not None:
            mask = resampleSize(mask, required_shape.tolist(), is_label=True)

        print("Image shape : {}".format(self.image.GetSize()))
        # Crop the image to (label_patch_size / num_down)
        self.image_down_shape = np.array(self.image.GetSize())
        self.patch_size = self.label_patch_size // 2**self.num_down
        _, _, self.z_length = self.image_down_shape - self.patch_size
        self.slide = self.patch_size // np.array((1, 1, self.overlap))
        total = self.z_length // self.slide[2] + 1
        self.patch_list = []
        self.patch_array_list = []
        with tqdm(total=total, desc="Clipping images...", ncols=60) as pbar:
            for z in range(0, self.z_length + 1, self.slide[2]):
                z_slice = slice(z, z + self.patch_size[2])
                if mask is not None:
                    patch_mask = sitk.GetArrayFromImage(mask[:, :, z_slice])
                    if (patch_mask == 0).all():
                        pbar.update(1)
                        continue

                patch = self.image[:, :, z_slice]
                patch.SetOrigin(self.image.GetOrigin())

                patch_array = sitk.GetArrayFromImage(patch)

                for _ in range(self.num_rep):
                    self.patch_list.append(patch)
                    self.patch_array_list.append(patch_array)
                
                pbar.update(1)

        
    def output(self, kind):
        if kind == "Array":
            return self.patch_array_list
        
        elif kind == "Image":
            return self.patch_list

        else:
            print("[ERROR] kind must be Array/Image")
            sys.exit()

    def restore(self, predict_array_list):
        segmented_array = np.zeros(self.image_down_shape[::-1])
        total = self.z_length // self.slide[2] + 1
        with tqdm(total=total, desc="Restoring images...", ncols=60) as pbar:
            assert len(predict_array_list) == (self.z_length // self.slide[2] + 1)
            for z, predict_array in zip(range(0, self.z_length + 1, self.slide[2]), predict_array_list):
                z_slice = slice(z, z + self.patch_size[2])
                segmented_array[z_slice, ...] = predict_array
                
                pbar.update(1)

        segmented = getImageWithMeta(segmented_array, self.image)
        segmented = resampleSize(segmented, self.image_shape, is_label=True)
        segmented = cropping(segmented, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())

        if (self.diff > 0).any():
            lower_crop_size = (self.diff // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in self.diff / 2]

            segmented = cropping(segmented, lower_crop_size, upper_crop_size)
        else:
            lower_pad_size = (abs(self.diff) // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in abs(self.diff) / 2]

            segmented = padding(segmented, lower_pad_size, upper_pad_size)

        return segmented

    def save(self, save_path, kind):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if not self.is_label:
            file_name = "image"
        else:
            file_name = "label"

        if kind == "Array":
            length = len(self.patch_array_list)
            with tqdm(total=length, desc="Saving image arrays...", ncols=60) as pbar:
                for i, patch_array in enumerate(self.patch_array_list):
                    path = save_path / "{}_{}.npy".format(file_name, str(i).zfill(3))
                    np.save(str(path), patch_array)
                    pbar.update(1)

        elif kind == "Image":
            length = len(self.patch_list)
            with tqdm(total=length, desc="Saving images...", ncols=60) as pbar:
                for i, patch in enumerate(self.patch_list):
                    path = save_path / "{}_{}.mha".format(file_name, str(i).zfill(3))
                    sitk.WriteImage(patch, str(path), True)
                    pbar.update(1)

        else:
            print("[ERROR] Kind must be Array/Image/Feature_map.")
            sys.exit()


