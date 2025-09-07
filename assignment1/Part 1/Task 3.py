import os
import cv2
import sys
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def compute_homography(
    src_pts: npt.NDArray[np.float32], dst_pts: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    a: list[list[float]] = []
    for (x, y), (X, Y) in zip(src_pts, dst_pts):
        a.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        a.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    A = np.array(a, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def warp_image(
    img: npt.NDArray[np.uint8],
    H: npt.NDArray[np.float64],
    out_shape: tuple[int, int],
    method: str = "nearest",
) -> npt.NDArray[np.uint8]:
    H_inv = np.linalg.inv(H)
    h_out, w_out = out_shape

    X, Y = np.meshgrid(np.arange(w_out), np.arange(h_out))
    ones = np.ones_like(X)
    coords = np.stack([X, Y, ones], axis=-1).reshape(-1, 3)

    src = coords @ H_inv.T
    src /= src[:, 2:3]
    x, y = src[:, 0], src[:, 1]

    channels = 1 if img.ndim == 2 else img.shape[2]
    warped = np.full((h_out, w_out, channels), 255, dtype=np.uint8)

    if method == "nearest":
        xi = np.round(x).astype(int)
        yi = np.round(y).astype(int)
        mask = (xi >= 0) & (xi < img.shape[1]) & (yi >= 0) & (yi < img.shape[0])
        warped.reshape(-1, channels)[mask] = img[yi[mask], xi[mask]].reshape(
            -1, channels
        )

    elif method == "bilinear":
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        dx = x - x0
        dy = y - y0
        mask = (x0 >= 0) & (x0 < img.shape[1] - 1) & (y0 >= 0) & (y0 < img.shape[0] - 1)

        c00 = img[y0[mask], x0[mask]]
        c10 = img[y0[mask], x0[mask] + 1]
        c01 = img[y0[mask] + 1, x0[mask]]
        c11 = img[y0[mask] + 1, x0[mask] + 1]

        dxm, dym = dx[mask][:, None], dy[mask][:, None]
        vals = (
            (1 - dxm) * (1 - dym) * c00
            + dxm * (1 - dym) * c10
            + (1 - dxm) * dym * c01
            + dxm * dym * c11
        )
        warped.reshape(-1, channels)[mask] = np.clip(vals, 0, 255).astype(np.uint8)

    return warped if img.ndim == 3 else warped[..., 0]

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(here, "autoLinear.png")
    logo_path = os.path.join(here, "iitlogo-20.jpg")

    if not os.path.exists(doc_path):
        print(f"Image file not found at {doc_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(logo_path):
        print(f"Image file not found at {logo_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(doc_path)
    logo = cv2.imread(logo_path)

    if img is None:
        sys.exit(1)

    if logo is None:
        sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB).astype(np.uint8)

    logo_gray = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    T = 240

    mask = (logo_gray < T).astype(np.uint8)
    mask = np.repeat(mask[:, :, None], 3, axis=2)

    doc_h, doc_w = img.shape[:2]
    new_w = int(0.2 * doc_w)
    scale = new_w / logo.shape[1]
    new_h = int(logo.shape[0] * scale)

    dst_pts = np.array(
        [[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]],
        dtype=np.float32,
    )
    src_pts = np.array(
        [
            [0, 0],
            [logo.shape[1] - 1, 0],
            [logo.shape[1] - 1, logo.shape[0] - 1],
            [0, logo.shape[0] - 1],
        ],
        dtype=np.float32,
    )

    H = compute_homography(src_pts, dst_pts)

    logo_resized = warp_image(logo, H, (new_h, new_w), method="bilinear")
    mask_resized = warp_image(mask, H, (new_h, new_w), method="nearest")

    # if logo_resized.shape[0] != new_h or logo_resized.shape[1] != new_w:
    #     logo_resized = cv2.resize(
    #         logo_resized, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    #     )
    # if mask_resized.shape[0] != new_h or mask_resized.shape[1] != new_w:
    #     mask_resized = cv2.resize(
    #         mask_resized, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    #     )

    y1, y2 = doc_h - new_h, doc_h
    x1, x2 = doc_w - new_w, doc_w

    roi = img[y1:y2, x1:x2].astype(np.float32)
    logo_f = logo_resized.astype(np.float32)
    mask_f = mask_resized.astype(np.float32)

    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    mask_f = np.clip(mask_f, 0.0, 1.0)

    blended = (1 - mask_f) * roi + mask_f * ((roi + logo_f) / 2)

    doc_out = img.copy()
    doc_out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    _ = plt.figure(figsize=(15, 6))

    ax_left = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax_mid_top = plt.subplot2grid((2, 3), (0, 1))
    ax_mid_bottom = plt.subplot2grid((2, 3), (1, 1))
    ax_right = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

    _ = ax_left.imshow(img)
    _ = ax_left.set_title("Original Document")
    _ = ax_left.axis("off")

    _ = ax_mid_top.imshow(logo_resized)
    _ = ax_mid_top.set_title("Resized Logo")
    _ = ax_mid_top.axis("off")

    _ = ax_mid_bottom.imshow(mask_resized[:, :, 0], cmap="gray")
    _ = ax_mid_bottom.set_title("Generated Mask")
    _ = ax_mid_bottom.axis("off")

    _ = ax_right.imshow(doc_out)
    _ = ax_right.set_title("Document with Logo (50% Blend)")
    _ = ax_right.axis("off")

    plt.tight_layout()
    plt.show()
    plt.imsave("resizedLogo.png", logo_resized)
    plt.imsave("maskLogo.png", mask_resized[:, :, 0], cmap="gray")
    plt.imsave("watermarkedDoc.png", doc_out)

if __name__ == "__main__":
    main()