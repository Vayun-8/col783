# from typing import tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

# from e import to_unit


def rgb_to_hsi(rgb_img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    hsi_img: npt.NDArray[np.float64] = np.zeros_like(rgb_img, dtype=np.float64)

    r = rgb_img[:, :, 0].astype(np.float64)
    g = rgb_img[:, :, 1].astype(np.float64)
    b = rgb_img[:, :, 2].astype(np.float64)

    intensity = np.mean(rgb_img, axis=2).astype(np.float64)
    min_rgb = np.min(rgb_img, axis=2).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = (
            1 - (3 / (np.sum(rgb_img, axis=2).astype(np.float64) + 1e-12)) * min_rgb
        )

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(np.clip(num / (den + 1e-12), -1, 1))

    hue = np.copy(theta)
    hue[b > g] = (2 * np.pi) - hue[b > g]
    hue = np.rad2deg(hue)
    hue[saturation < 1e-8] = 0

    hsi_img[:, :, 0] = hue
    hsi_img[:, :, 1] = saturation
    hsi_img[:, :, 2] = intensity
    return hsi_img


def hsi_to_rgb(hsi_img: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    h = hsi_img[:, :, 0].astype(np.float32)
    s = hsi_img[:, :, 1].astype(np.float32)
    i = hsi_img[:, :, 2].astype(np.float32)
    h = np.deg2rad(h)

    rgb_img: npt.NDArray[np.float32] = np.zeros_like(hsi_img, dtype=np.float32)
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    idx = (h >= 0) & (h < 2 * np.pi / 3)
    b[idx] = i[idx] * (1 - s[idx])
    r[idx] = i[idx] * (
        1 + (s[idx] * np.cos(h[idx])) / (np.cos(np.pi / 3 - h[idx]) + 1e-12)
    )
    g[idx] = 3 * i[idx] - (r[idx] + b[idx])

    idx = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h_shifted = h - (2 * np.pi / 3)
    r[idx] = i[idx] * (1 - s[idx])
    g[idx] = i[idx] * (
        1
        + (s[idx] * np.cos(h_shifted[idx]))
        / (np.cos(np.pi / 3 - h_shifted[idx]) + 1e-12)
    )
    b[idx] = 3 * i[idx] - (r[idx] + g[idx])

    idx = (h >= 4 * np.pi / 3) & (h <= 2 * np.pi)
    h_shifted = h - (4 * np.pi / 3)
    g[idx] = i[idx] * (1 - s[idx])
    b[idx] = i[idx] * (
        1
        + (s[idx] * np.cos(h_shifted[idx]))
        / (np.cos(np.pi / 3 - h_shifted[idx]) + 1e-12)
    )
    r[idx] = 3 * i[idx] - (g[idx] + b[idx])

    rgb_img[:, :, 0] = r
    rgb_img[:, :, 1] = g
    rgb_img[:, :, 2] = b
    return np.clip(rgb_img, 0, 1)


def create_rgb_mask(
    rgb_image: npt.NDArray[np.float32],
    seed_point: tuple[int, int],
    tolerances: tuple[float, float, float],
) -> npt.NDArray[np.uint8]:
    y_s, x_s = seed_point
    r_tol, g_tol, b_tol = tolerances

    seed_color = rgb_image[y_s, x_s]
    r_s, g_s, b_s = seed_color

    r_min, r_max = r_s - r_tol, r_s + r_tol
    g_min, g_max = g_s - g_tol, g_s + g_tol
    b_min, b_max = b_s - b_tol, b_s + b_tol

    mask = (
        (rgb_image[:, :, 0] >= r_min)
        & (rgb_image[:, :, 0] <= r_max)
        & (rgb_image[:, :, 1] >= g_min)
        & (rgb_image[:, :, 1] <= g_max)
        & (rgb_image[:, :, 2] >= b_min)
        & (rgb_image[:, :, 2] <= b_max)
    )
    return mask.astype(np.uint8)


def create_hsi_mask(
    hsi_image: npt.NDArray[np.float64],
    seed_point: tuple[int, int],
    tolerances: tuple[float, float, float],
) -> npt.NDArray[np.uint8]:
    y_s, x_s = seed_point
    h_tol, s_tol, i_tol = tolerances

    H, S, I = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]
    h_s, s_s, i_s = hsi_image[y_s, x_s]

    h_min, h_max = h_s - h_tol, h_s + h_tol
    if h_min < 0:
        hue_mask = (H >= (360 + h_min)) | (H <= h_max)
    elif h_max > 360:
        hue_mask = (H >= h_min) | (H <= (h_max - 360))
    else:
        hue_mask = (H >= h_min) & (H <= h_max)

    s_min, s_max = max(0, s_s - s_tol), min(1.0, s_s + s_tol)
    i_min, i_max = max(0, i_s - i_tol), min(1.0, i_s + i_tol)

    sat_mask = (S >= s_min) & (S <= s_max)
    int_mask = (I >= i_min) & (I <= i_max)
    mask = hue_mask & sat_mask & int_mask
    return mask.astype(np.uint8)


def apply_hsi_target(
    hsi_img: npt.NDArray[np.float64],
    mask: npt.NDArray[np.uint8],
    source_hsi: tuple[float, float, float],
    target_hsi: tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    result = hsi_img.copy().astype(np.float64)
    masked = mask.astype(bool)

    h_s, s_s, i_s = source_hsi
    h_t, s_t, i_t = target_hsi

    dH = (h_t - h_s) % 360.0
    s_factor = (s_t / s_s) if s_s > 1e-6 else 1.0
    i_factor = (i_t / i_s) if i_s > 1e-6 else 1.0

    result[..., 0][masked] = (result[..., 0][masked] + dH) % 360.0
    result[..., 1][masked] = np.clip(result[..., 1][masked] * s_factor, 0.0, 1.0)
    result[..., 2][masked] = np.clip(result[..., 2][masked] * i_factor, 0.0, 1.0)

    return result


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "test_image.jpg")

    if not os.path.exists(path):
        print(f"Image file not found at {path}", file=sys.stderr)
        sys.exit(1)

    img = mpimg.imread(path)
    print(type(img[0][0][0]))

    hsi = rgb_to_hsi(img)

    H_vis = hsi[:, :, 0] / 360.0
    S_vis = hsi[:, :, 1]
    I_vis = hsi[:, :, 2]

    # plt.figure()
    # plt.imshow(img)
    # pts = plt.ginput(1, timeout=0)
    #
    # print(pts)

    _, axs = plt.subplots(1, 4)
    axs[0].imshow(img)
    axs[0].set_title("RGB")
    axs[0].axis("off")

    axs[1].imshow(H_vis, cmap="hsv")
    axs[1].set_title("Hue")
    axs[1].axis("off")

    axs[2].imshow(S_vis, cmap="gray")
    axs[2].set_title("Saturation")
    axs[2].axis("off")

    axs[3].imshow(I_vis, cmap="gray")
    axs[3].set_title("Intensity")
    axs[3].axis("off")
    # pts = plt.ginput(1, timeout=0)
    #
    # print(pts)
    plt.tight_layout()
    plt.imsave("hue_channel.png", H_vis, cmap="hsv")
    plt.imsave("saturation_channel.png", S_vis, cmap="gray")
    plt.imsave("intensity_channel.png", I_vis, cmap="gray")
    plt.show()

    seed_point = (131, 287)
    print(img[seed_point])
    print(np.min(img))
    # rgb_tolerances = (0.80, 0.80, 0.80)
    rgb_tolerances = (0.2, 0.3, 0.2)
    hsi_tolerances = (28.0, 0.7, 0.4)

    rgb_mask = create_rgb_mask(img, seed_point, rgb_tolerances)
    hsi_mask = create_hsi_mask(hsi, seed_point, hsi_tolerances)

    _, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title("RGB")
    axs[0].axis("off")

    axs[1].imshow(rgb_mask, cmap="gray")
    axs[1].set_title("RGB Mask")
    axs[1].axis("off")

    axs[2].imshow(hsi_mask, cmap="gray")
    axs[2].set_title("Hsi Mask")
    axs[2].axis("off")
    plt.imsave("rgb_mask.png", rgb_mask, cmap="gray")
    plt.imsave("hsi_mask.png", hsi_mask, cmap="gray")
    plt.show()

    target = (340.0, 0.6, 0.8)

    tf = apply_hsi_target(hsi, hsi_mask, hsi[seed_point], target)

    # plt.figure()
    # plt.imshow(hsi_to_rgb(tf))
    # plt.show()
    _, axs = plt.subplots(1, 2)

    axs[0].imshow(img)
    axs[0].set_title("RGB")
    axs[0].axis("off")

    axs[1].imshow(hsi_to_rgb(tf))
    axs[1].set_title("Transformed Image")
    axs[1].axis("off")
    plt.imsave("transformed_rgb.png", hsi_to_rgb(tf))
    plt.show()


if __name__ == "__main__":
    main()