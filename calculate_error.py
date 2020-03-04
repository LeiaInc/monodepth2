import os
import glob
import cv2
import numpy as np
import re

def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)

def load_mask(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.logical_and(mask > 254, mask < 256)

def load_numpy(disp_path):
    return np.load(disp_path).squeeze()

def load_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #
    img = np.reshape(dispariy, newshape=(height, width))
    # img[img==np.inf] = 0
    # img = np.flipud(img).astype('uint8')
    # #
    # show(img, "disparity")

    return img

def normalize_disparity(disp):
    min_dp = np.amin(disp)
    max_dp = np.amax(disp)

    return (disp - min_dp) / (max_dp - min_dp) + 0.01 # Assume 0.01 to be the minimum disp


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


def calculate_error(pred, gt, confidence_mask):
    # mask out based on confidence map
    gt = gt[confidence_mask]
    pred = pred[confidence_mask]

    value_mask = np.logical_and(gt > 0, gt < 70)
    gt = gt[value_mask]
    pred = pred[value_mask]

    # min, max normalization
    pred = normalize_disparity(pred)

    pred *= np.median(gt) / np.median(pred)
    errors = compute_errors(gt, pred)

    return errors

mask_dir = "/home/owenhua/Devel/monodepth2/middlebury/left-masks"
assert_dir = "/home/owenhua/Devel/monodepth2/middlebury/left-images/"
gt_dir = "/home/owenhua/Devel/monodepth2/middlebury/left-depths"

pred_paths = sorted(glob.glob(assert_dir + '/*holopix_no_pt_full.npy'))
mask_paths = sorted(glob.glob(mask_dir + '/*'))
gt_paths = sorted(glob.glob(gt_dir + '/*'))

e1_list, e2_list, e3_list, e4_list = [], [], [], []

for id in range(len(pred_paths)):

    # print(pred_paths[id])
    # print(mask_paths[id])
    # print(gt_paths[id])

    pred = load_numpy(pred_paths[id])
    mask = load_mask(mask_paths[id])
    gt = load_pfm(gt_paths[id])

    e1, e2, e3, e4 = calculate_error(pred, gt, mask)
    e1_list.append(e1)
    e2_list.append(e2)
    e3_list.append(e3)
    e4_list.append(e4)

print(np.mean(np.array(e1_list)))
print(np.mean(np.array(e2_list)))
print(np.mean(np.array(e3_list)))
print(np.mean(np.array(e4_list)))

# colored_portion = cv2.bitwise_or(disp, disp, mask = mask)


# cv2.imshow('img', colored_portion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
