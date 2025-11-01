"""
fMRI ROI Extraction and Normalization Script
--------------------------------------------
This script extracts region-of-interest (ROI) time series from fMRI dtseries files,
performs voxel-wise normalization, and saves ROI-masked data for downstream analysis.

Dependencies:
    - nibabel
    - numpy
    - pandas
    - hcp_utils  (custom module providing HCP MMP atlas utilities)

Author: Jianxiong Gao
License: Apache 2.0
"""

import nibabel as nib
from pathlib import Path
import numpy as np
import pandas as pd
import hcp_utils

# Template CIFTI file with correct spatial axes
TEMPLATE_ZERO = "/mnt/test/user/gaojianxiong/data/brain_vis/temp.zeros.dscalar.nii"


def save_to_dscalar(template_data, fname, template_zero_file=TEMPLATE_ZERO, axis_name=None):
    """
    Save a 2D array as a CIFTI .dscalar file, using a reference template to preserve geometry.

    Args:
        template_data (ndarray): 2D array of shape (n_scalars, n_vertices)
        fname (str): Output file path
        template_zero_file (str): Path to a reference .dscalar file
        axis_name (list[str] | None): Optional names for scalar dimension
    """
    hcp = nib.load(template_zero_file)
    assert len(template_data.shape) == 2, "Input data must be 2D."

    if axis_name is None:
        axis = nib.cifti2.cifti2_axes.ScalarAxis([str(x) for x in range(template_data.shape[0])])
    else:
        axis = nib.cifti2.cifti2_axes.ScalarAxis(axis_name)

    header = nib.cifti2.cifti2.Cifti2Header.from_axes((axis, hcp.header.get_axis(1)))
    new_img = nib.cifti2.cifti2.Cifti2Image(
        np.array(template_data).copy(),
        header,
        hcp.nifti_header,
        hcp.extra,
        hcp.file_map,
    )

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    new_img.to_filename(Path(fname))


# -------------------------------------------------------------------------
# Define ROI index lists (customized from the HCP MMP1.0 atlas)
# -------------------------------------------------------------------------
ALL_ROI_IDS = [
    # Parietal, temporal, and occipital clusters
    8, 16, 17, 29, 30, 42, 45, 46, 47, 143, 152,
    24, 123, 128, 129, 130, 139, 140, 141, 303, 308, 309, 310, 356,
    143, 152, 343,
    # Right hemisphere mirror (+180)
    *[i + 180 for i in [8, 16, 17, 29, 30, 42, 45, 46, 47, 143, 152,
                        24, 123, 128, 129, 130, 139, 140, 141, 303, 308, 309, 310, 343, 356]],
    # Prefrontal cortex
    *[i for i in [63, 64, 66, 67, 68, 71, 72, 73, 74, 75, 84, 87, 88, 345]],
    *[i + 180 for i in [63, 64, 66, 67, 68, 71, 72, 73, 74, 75, 84, 87, 88, 345]],
    # Language network enhancement (Broca/Wernicke areas)
    74, 75, 254, 255, 128, 308, 24, 204, 123, 303, 356,
]

ROI_NAMES = [
    'V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'LO1', 'LO2', 'LO3', 'PIT',
    'V4t', 'V6', 'V6A', 'V7', 'V8', 'PH', 'FFC', 'IP0', 'MT', 'MST', 'FST',
    'VVC', 'VMV1', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3'
]

# Initialize ROI mask
roi_mask = np.zeros_like(hcp_utils.mmp.map_all, dtype=bool)
rois = {}

# Build ROI masks from MMP labels
for name in ROI_NAMES:
    label_name = f"L_{name}"
    for k, v in hcp_utils.mmp.labels.items():
        if v == label_name:
            idx = k
    rois[name] = np.where(
        (hcp_utils.mmp.map_all == idx) | (hcp_utils.mmp.map_all == (180 + idx))
    )[0]
    roi_mask[rois[name]] = True

print(f"Total ROI voxels (visual cortex subset): {roi_mask.sum()}")

# Additional ROI selection based on ID list
roi_indices = []
for roi_id in ALL_ROI_IDS:
    if roi_id in hcp_utils.mmp.labels:
        roi_indices.append(roi_id)

for i in roi_indices:
    rois[i] = np.where(hcp_utils.mmp.map_all == i)[0]
    roi_mask[rois[i]] = True

# -------------------------------------------------------------------------
# fMRI time-series extraction and normalization
# -------------------------------------------------------------------------
all_data = []
for run_id in range(1, 21):
    beh_path = f"/ssd/gaojianxiong/CineBrain/sub_0001/beh_data/sub_0001_{run_id}.csv"
    beh_df = pd.read_csv(beh_path)
    start_time = beh_df["videoStartTime"].values[0] + 4
    end_time = beh_df["videoStopTime"].values[0] + 4
    print(f"Run {run_id}: frames {start_time/0.8:.1f} to {end_time/0.8:.1f}")

    nii_path = (
        f"/ssd/gaojianxiong/CineBrain/sub_0001/dtseries/"
        f"sub-0001_task-shape_run-{run_id}_space-fsLR_den-91k_bold.dtseries.nii"
    )
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    all_data.append(data[int(start_time / 0.8) + 1 : int(end_time / 0.8) + 1, :])

fmri_data = np.concatenate(all_data, axis=0)

# Voxel-wise z-normalization
mean = np.mean(fmri_data, axis=0)
std = np.std(fmri_data, axis=0, ddof=1)
fmri_data = (fmri_data - mean) / std
fmri_data[np.isnan(fmri_data)] = 0

# -------------------------------------------------------------------------
# ROI masking and data export
# -------------------------------------------------------------------------
np.save("roi_var_visual.npy", roi_mask)
fmri_data_roi = fmri_data[:, roi_mask]
print(f"fMRI data shape after ROI selection: {fmri_data_roi.shape}")

output_dir = Path("/ssd/gaojianxiong/CineBrain/sub_0001/visual_audio")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(len(fmri_data_roi)):
    np.save(output_dir / f"{i}.npy", fmri_data_roi[i])
