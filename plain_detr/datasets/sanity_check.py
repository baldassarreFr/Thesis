import logging

import torch
from torch.utils.data import DataLoader

import plain_detr.util.misc as utils


def sanity_check_logger(dataset_train, dataset_val, args):
    """Sanity check: print shapes for one batch from train and val loaders."""
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SANITY CHECK - Tensor Shapes & Coordinate Alignments")
    logger.info("=" * 60)

    # Helper function per leggere la shape a prescindere dal tipo di dato
    def get_img_shape(img_batch):
        if hasattr(img_batch, "tensors"):  # È un NestedTensor di DETR
            return img_batch.tensors.shape
        return img_batch.shape  # È un tensore normale

    # --- TRAIN BATCH ---
    train_iter = iter(
        DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=0)
    )
    train_img, train_target = next(train_iter)

    logger.info("TRAIN batch (Expect Random Crop 800x800):")
    logger.info(f"  Image shape:      {get_img_shape(train_img)}")
    logger.info(f"  Target orig_size: {train_target[0]['orig_size'].tolist()} (Anchor: Fixed Crop)")
    logger.info(f"  Target size:      {train_target[0]['size'].tolist()} (Dynamic output)")
    logger.info(f"  Target boxes:     {train_target[0]['boxes'].shape}")
    logger.info(f"  Target labels:    {train_target[0]['labels'].shape}")
    logger.info("-" * 60)

    # --- VAL BATCH ---
    val_iter = iter(DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=0))
    val_img, val_target = next(val_iter)

    logger.info("VAL batch (Expect Proportional Resize max 2000):")
    logger.info(f"  Image shape:      {get_img_shape(val_img)}")
    logger.info(f"  Target orig_size: {val_target[0]['orig_size'].tolist()} (Anchor: Fixed Crop)")
    logger.info(f"  Target size:      {val_target[0]['size'].tolist()} (Dynamic output)")
    logger.info(f"  Target boxes:     {val_target[0]['boxes'].shape}")
    logger.info(f"  Target labels:    {val_target[0]['labels'].shape}")
    logger.info("=" * 60)
