import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta
from downSamplePatchCreater import DownSamplePatchCreater
from pathlib import Path
from tqdm import tqdm
import torch
import cloudpickle
import re


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--label_patch_size", help="512-512-32", default="512-512-32")
    parser.add_argument("--plane_size", help="512-512", default="512-512")
    parser.add_argument("--overlap", help="1", default=1, type=int)
    parser.add_argument("--num_down", help="2", default=2, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    image = sitk.ReadImage(args.image_path)

    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.label_patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.label_patch_size))
        sys.exit()

    label_patch_size = [int(s) for s in matchobj.groups()]

    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)", args.plane_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.plane_size))
        sys.exit()

    plane_size = [int(s) for s in matchobj.groups()]

    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    dspc = DownSamplePatchCreater(
            image = image,
            label_patch_size = label_patch_size,
            plane_size = plane_size,
            overlap = args.overlap,
            num_down = args.num_down,
            is_label = False,
            mask = mask
            )
    dspc.execute()
    image_array_list = dspc.output("Array")

    """ Load model. """
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    segmented_array_list = []
    total = len(image_array_list)
    with tqdm(total = total, desc="Segmenting images...", ncols=60) as pbar:
        for image_array in image_array_list:
            image_array = torch.from_numpy(image_array).to(device, dtype=torch.float)
            image_array = image_array[None, None, ...]

            segmented_array = model(image_array)
            segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
            segmented_array = np.squeeze(segmented_array)
            segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)

            segmented_array_list.append(segmented_array)
            pbar.update(1)

    """ Restore module. """
    segmented = dspc.restore(segmented_array_list)

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)



if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
