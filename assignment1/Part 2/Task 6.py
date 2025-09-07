import os
import sys
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
from typing import Literal

def to_uint8(img_f: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    out = np.clip(img_f * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return out

def conv2d_raw(
    input_arr: npt.NDArray[np.float32],
    kernel: npt.NDArray[np.float32],
    c: float = 0.0,
    mode: Literal["same", "full"] = "same",
    flip_kernel: bool = True,
) -> npt.NDArray[np.float32]:
    a = np.asarray(input_arr, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)

    if flip_kernel:
        k = np.flip(k, (0, 1))

    H, W = a.shape
    kh, kw = k.shape

    if mode == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
        out_h = H
        out_w = W
    elif mode == "full":
        pad_h = kh - 1
        pad_w = kw - 1
        padded = np.pad(
            a, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0
        )
        out_h = H + kh - 1
        out_w = W + kw - 1

    out = np.empty((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        block = padded[i : i + kh, :]
        row_acc = np.zeros(out_w, dtype=np.float32)
        for r in range(kh):
            conv_r = np.convolve(block[r], k[r], mode="valid")
            row_acc += conv_r.astype(np.float32)
        out[i, :] = row_acc

    if c != 0.0:
        out = out + np.float32(c)

    return out

def disk_kernel(radius: int, dtype: type = np.float32) -> npt.NDArray[np.float32]:
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    mask = (xx * xx + yy * yy) <= (r * r)
    k = mask.astype(np.float32)
    s = k.sum()
    if s == 0:
        k[r, r] = 1.0
        s = 1.0
    k /= s
    return k.astype(dtype)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "memorial.hdr")

    if not os.path.exists(path):
        print(f"Image file not found at {path}", file=sys.stderr)
        sys.exit(1)
    GAMMA = 0.1
    RADIUS = 7

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Image not found: {path}")
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    w = disk_kernel(RADIUS, dtype=np.float32)
    A_f = conv2d_raw((raw**GAMMA).astype(np.float32), w)
    B_f = conv2d_raw(raw.astype(np.float32), w)
    img = to_uint8(A_f)
    img2 = to_uint8(B_f**GAMMA)

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("T'(w * f)")
    axs[0].axis("off")

    axs[1].imshow(img2, cmap="gray")
    axs[1].set_title("w * T'(f)")
    axs[1].axis("off")

    # _ = plt.figure(figsize=(12, 6))
    # _ = plt.subplot(1, 2, 1)
    # _ = plt.title("Original (linear)")
    # _ = plt.imshow(img, cmap="gray")
    #
    # _ = plt.subplot(1, 2, 2)
    # _ = plt.title("Original (linear)")
    # _ = plt.imshow(img2, cmap="gray")
    # plt.tight_layout()
    cv2.imwrite("gamma_after_convolution.png", img)
    cv2.imwrite("gamma_before_convolution.png", img2)
    plt.show()


if __name__ == "__main__":
    main()
