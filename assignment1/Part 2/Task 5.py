import os
import sys
import time
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
from typing import Literal


def to_unit(img_u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    return (img_u8.astype(np.float32) / 255.0).astype(np.float32)


def to_uint8(img_f: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    out = np.clip(img_f * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return out


def conv1d_raw(
    signal: npt.NDArray[np.float32], kernel: npt.NDArray[np.float32], c: float = 0.0
) -> npt.NDArray[np.float32]:
    sig = signal.astype(np.float32)
    k = kernel.astype(np.float32)
    m = k.shape[0]
    pad = m // 2
    padded = np.pad(sig, (pad, pad), mode="edge")
    out = np.zeros_like(sig, dtype=np.float32)
    for i in range(m):
        out += k[i] * padded[i : i + sig.shape[0]]
    if c != 0.0:
        out = out + np.float32(c)
    return out


def conv1d(
    signal: npt.NDArray[np.uint8], kernel: npt.NDArray[np.float32], c: float = 0.0
) -> npt.NDArray[np.uint8]:
    sig_f = to_unit(signal)
    out_f = conv1d_raw(sig_f, kernel, c=c)
    return to_uint8(out_f)


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


def conv2d(
    image: npt.NDArray[np.uint8], kernel: npt.NDArray[np.float32], c: float = 0.0
) -> npt.NDArray[np.uint8]:
    img_f = to_unit(image)
    out_f = conv2d_raw(img_f, kernel, c=c)
    return to_uint8(out_f)


def gaussian_kernel_1d(sigma: float) -> npt.NDArray[np.float32]:
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    g /= g.sum()
    return g.astype(np.float32)


def gaussian_kernel_2d(sigma: float) -> npt.NDArray[np.float32]:
    g1 = gaussian_kernel_1d(sigma)
    return np.outer(g1, g1).astype(np.float32)


def separable_conv2d_via_conv1d_raw(
    image: npt.NDArray[np.float32], g1: npt.NDArray[np.float32], c: float = 0.0
) -> npt.NDArray[np.float32]:
    img = image.astype(np.float32)
    H, W = img.shape
    tmp = np.empty((H, W), dtype=np.float32)
    for r in range(H):
        tmp[r, :] = conv1d_raw(img[r, :], g1, c=0.0)
    out = np.empty((H, W), dtype=np.float32)
    for col in range(W):
        out[:, col] = conv1d_raw(tmp[:, col], g1, c=0.0)
    if c != 0.0:
        out = out + np.float32(c)
    return out


def separable_conv2d_via_conv1d(
    image: npt.NDArray[np.uint8], g1: npt.NDArray[np.float32], c: float = 0.0
) -> npt.NDArray[np.uint8]:
    img_f = to_unit(image)
    out_f = separable_conv2d_via_conv1d_raw(img_f, g1, c=c)
    return to_uint8(out_f)


def normalize_for_display(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    y = x - float(np.nanmin(x))
    mx = float(np.nanmax(y))
    if mx > 0:
        y = y / mx
    return y.astype(np.float32)


def laplacian(img: npt.NDArray[np.uint8]):
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    c_uint8 = 128.0
    c_scaled = float(c_uint8 / 255.0)

    out_u8 = conv2d(img, lap, c=c_scaled)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title("Original (grayscale)")
    axs[0].axis("off")

    axs[1].imshow(out_u8, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Laplacian + c={int(c_uint8)}")
    axs[1].axis("off")

    plt.tight_layout()
    cv2.imwrite("output_laplacian_original.png", img)
    cv2.imwrite("output_laplacian_result.png", out_u8)
    plt.show()


def gaussian(img: npt.NDArray[np.uint8], sigma: float = 8.0) -> None:
    g1 = gaussian_kernel_1d(sigma)
    k2 = gaussian_kernel_2d(sigma)

    t0 = time.perf_counter()
    out_naive = conv2d(img, k2, c=0.0)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    out_sep = separable_conv2d_via_conv1d(img, g1, c=0.0)
    t3 = time.perf_counter()

    naive_time = t1 - t0
    sep_time = t3 - t2

    print(f"image shape: {img.shape}")
    print(f"sigma: {sigma}, kernel size: {k2.shape}")
    print(f"naive conv2d time: {naive_time:.4f} s")
    print(f"separable conv time: {sep_time:.4f} s")

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(out_naive, cmap="gray")
    axs[1].set_title(f"Naive 2D ({naive_time:.3f}s)")
    axs[1].axis("off")

    axs[2].imshow(out_sep, cmap="gray")
    axs[2].set_title(f"Separable ({sep_time:.3f}s)")
    axs[2].axis("off")

    fig.tight_layout()
    cv2.imwrite("output_gaussian_original.png", img)
    cv2.imwrite("output_gaussian_naive.png", out_naive)
    cv2.imwrite("output_gaussian_separable.png", out_sep)

    plt.show()


def cong(img: npt.NDArray[np.uint8], sigma: float = 8.0) -> None:
    img_f = to_unit(img)

    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    g = gaussian_kernel_2d(sigma).astype(np.float32)

    # 1) L * (G * f)
    t0 = time.perf_counter()
    g_conv = conv2d_raw(img_f, g)  # G * f
    res1_f = conv2d_raw(g_conv, lap)  # L * (G*f)
    t1 = time.perf_counter()

    # 2) G * (L * f)
    t2 = time.perf_counter()
    l_conv = conv2d_raw(img_f, lap, mode="same")  # L * f
    res2_f = conv2d_raw(l_conv, g, mode="same")  # G * (L*f)
    t3 = time.perf_counter()

    # 3) (L * G) * f
    t4 = time.perf_counter()
    LoG_kernel = conv2d_raw(g, lap, mode="full")  # (G * L)
    res3_f = conv2d_raw(img_f, LoG_kernel, mode="same", flip_kernel=True)
    t5 = time.perf_counter()

    print(f"L*(G*f): {(t1 - t0):.6f}s")
    print(f"G*(L*f): {(t3 - t2):.6f}s")
    print(f"(L*G)*f: {(t5 - t4):.6f}s")

    diff12 = float(np.max(np.abs(res1_f - res2_f)))
    diff13 = float(np.max(np.abs(res1_f - res3_f)))
    diff23 = float(np.max(np.abs(res2_f - res3_f)))
    print(f"max abs diff res1 vs res2: {diff12:.6e}")
    print(f"max abs diff res1 vs res3: {diff13:.6e}")
    print(f"max abs diff res2 vs res3: {diff23:.6e}")

    d1 = normalize_for_display(res1_f)
    d2 = normalize_for_display(res2_f)
    d3 = normalize_for_display(res3_f)

    disp_res1 = to_uint8(d1)
    disp_res2 = to_uint8(d2)
    disp_res3 = to_uint8(d3)

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(disp_res1, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title("L*(G*f)")
    axs[1].axis("off")

    axs[2].imshow(disp_res2, cmap="gray", vmin=0, vmax=255)
    axs[2].set_title("G*(L*f)")
    axs[2].axis("off")

    axs[3].imshow(disp_res3, cmap="gray", vmin=0, vmax=255)
    axs[3].set_title("(L*G)*f")
    axs[3].axis("off")

    fig.tight_layout()
    cv2.imwrite("output_cong_original.png", img)
    cv2.imwrite("output_cong_lgf.png", disp_res1)
    cv2.imwrite("output_cong_glf.png", disp_res2)
    cv2.imwrite("output_cong_lg_f.png", disp_res3)
    plt.show()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "test_image.jpg")

    if not os.path.exists(path):
        print(f"Image file not found at {path}", file=sys.stderr)
        sys.exit(1)

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit(1)

    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    img = raw.astype(np.uint8)

    laplacian(img)

    gaussian(img, 8.0)

    cong(img, 8.0)


# if __name__ == "__main__":
#     cong()
