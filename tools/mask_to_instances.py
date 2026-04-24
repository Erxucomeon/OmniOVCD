import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def binary_mask_to_instance_masks(
    mask_0_255: np.ndarray,
    *,
    connectivity: int = 2,
    min_area: int = 0,
    sort_by_area: bool = True,
) -> list[np.ndarray]:
    if mask_0_255 is None:
        raise ValueError("mask_0_255 is None")

    mask_np = np.asarray(mask_0_255)
    if mask_np.ndim != 2:
        raise ValueError(f"mask_0_255 must be 2D [H, W], got shape={mask_np.shape}")

    fg = (mask_np != 0).astype(np.uint8)

    labeled = measure.label(fg, connectivity=connectivity, background=0)
    num = int(labeled.max())

    instances: list[tuple[int, np.ndarray]] = []
    for instance_id in range(1, num + 1):
        inst = (labeled == instance_id)
        area = int(inst.sum())
        if min_area > 0 and area < min_area:
            continue
        inst_u8 = (inst.astype(np.uint8) * 255)
        instances.append((area, inst_u8))

    if sort_by_area:
        instances.sort(key=lambda x: x[0], reverse=True)

    return [m for _, m in instances]


def export_instances_from_mask_path(
    mask_path: str,
    out_dir: str,
    *,
    connectivity: int = 2,
    min_area: int = 0,
    sort_by_area: bool = True,
    prefix: str | None = None,
) -> list[str]:
    """Load a 0/255 mask image (including .tif) and export each instance as a PNG.

    Returns:
        List[str]: output png paths.
    """
    mask_path = str(mask_path)
    out_dir = str(out_dir)

    img = Image.open(mask_path)
    # WHU-CD label tif is typically single-channel; ensure we end up with 2D.
    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")

    mask_np = np.array(img)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # normalize to 0/255 (treat any non-zero as 255)
    mask_0_255 = np.where(mask_np != 0, 255, 0).astype(np.uint8)

    inst_masks = binary_mask_to_instance_masks(
        mask_0_255,
        connectivity=connectivity,
        min_area=min_area,
        sort_by_area=sort_by_area,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(mask_path).stem
    if prefix is None:
        prefix = stem

    out_paths: list[str] = []
    for i, m in enumerate(inst_masks):
        out_path = str(Path(out_dir) / f"{prefix}_{i:04d}.png")
        Image.fromarray(m, mode="L").save(out_path)
        out_paths.append(out_path)

    return out_paths


def main():
    default_in = "./dataset/WHU-CD/test_256/label2/441_3.tif"
    default_out = "./tools/t2"

    in_path = os.environ.get("MASK_PATH", default_in)
    out_dir = os.environ.get("OUT_DIR", default_out)

    export_instances_from_mask_path(in_path, out_dir, connectivity=2)


if __name__ == "__main__":
    main()
