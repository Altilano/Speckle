import numpy as np
import warnings


def _moments(w, x0=0, y0=0):
    """Center of mass and standard deviations of a 2D weight array.
    (x0, y0) is the pixel index of w[0, 0] in the full image."""
    total = w.sum()
    if total <= 0:
        raise ValueError("No signal found (sum of intensities <= 0).")
    x = np.arange(w.shape[1], dtype=np.float64) + x0
    y = np.arange(w.shape[0], dtype=np.float64) + y0
    px = w.sum(axis=0) / total          # marginal distributions
    py = w.sum(axis=1) / total
    cx = np.sum(x * px)
    cy = np.sum(y * py)
    sx = np.sqrt(max(np.sum((x - cx) ** 2 * px), 0.0))
    sy = np.sqrt(max(np.sum((y - cy) ** 2 * py), 0.0))
    return cx, cy, sx, sy


def beam_std(image, pixel_size=None, background="auto", n_sigma=4.0, n_iter=10, border=10):
    """Spatial standard deviation of an intensity image about its center of mass.

    Treats the image as a 2D distribution I(x, y) and computes, with I as weight,
        cx = sum(x I) / sum(I),     sigma_x^2 = sum((x - cx)^2 I) / sum(I)
    (same for y). Works for any beam profile, Gaussian or not.

    For a Gaussian beam I ~ exp(-2 r^2 / w^2): sigma_x = w / 2, i.e. w = 2 * sigma
    (the "D4sigma" beam diameter is 4 * sigma).

    Why the integration window: second moments weight pixels by distance^2, so
    camera noise far from the beam badly inflates sigma. After subtracting the
    background, the moments are therefore computed only inside a window of
    +/- n_sigma * sigma around the centroid, iterated until it converges (as in
    ISO 11146). With n_sigma = 4 the truncation error for a Gaussian is ~0.1 %.

    Parameters
    ----------
    image : 2D array      Camera image.
    pixel_size : float    Pixel pitch (any length unit). If given, outputs are in that
                          unit; otherwise in pixels.
    background : "auto", float, 2D array or None
        "auto" -> mean of a frame of `border` pixels at the image edges
                  (assumes the beam does not reach the edges);
        float  -> constant offset to subtract;
        array  -> dark frame, same shape as `image`;
        None   -> no subtraction.
    n_sigma : float or None
        Half-width of the integration window in units of sigma. None = use the whole
        image in one pass (only for clean, zero-background images).
    n_iter : int          Max iterations of the window update.
    border : int          Width of the edge frame for the "auto" background.

    Returns
    -------
    dict with
        "cx", "cy"             center of mass (origin = center of pixel [0, 0])
        "sigma_x", "sigma_y"   std along x (columns) and y (rows)
        "sigma_r"              sqrt(sigma_x^2 + sigma_y^2), the rms radius
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    if np.issubdtype(img.dtype, np.integer) and img.max() >= np.iinfo(img.dtype).max:
        warnings.warn("Image is saturated; sigma will be underestimated.")

    img = img.astype(np.float64)

    # --- background subtraction (no clipping: zero-mean noise averages out) ---
    if isinstance(background, str):
        if background != "auto":
            raise ValueError('background must be "auto", a number, an array or None.')
        b = max(1, min(border, img.shape[0] // 4, img.shape[1] // 4))
        edge = np.concatenate([
            img[:b, :].ravel(), img[-b:, :].ravel(),
            img[b:-b, :b].ravel(), img[b:-b, -b:].ravel(),
        ])
        img = img - edge.mean()  # mean (not median): unbiased even with 8-bit quantization
    elif background is not None:
        img = img - np.asarray(background, dtype=np.float64)

    if n_sigma is None:
        cx, cy, sx, sy = _moments(img)
    else:
        # Seed: only pixels above 10 % of the peak, to get a first centroid/size
        seed = np.where(img >= 0.1 * img.max(), img, 0.0)
        cx, cy, sx, sy = _moments(seed)

        H, W = img.shape
        for _ in range(n_iter):
            hx = max(n_sigma * sx, 2.0)
            hy = max(n_sigma * sy, 2.0)
            x0, x1 = max(int(round(cx - hx)), 0), min(int(round(cx + hx)) + 1, W)
            y0, y1 = max(int(round(cy - hy)), 0), min(int(round(cy + hy)) + 1, H)
            new = _moments(img[y0:y1, x0:x1], x0, y0)
            done = max(abs(a - b) for a, b in zip(new, (cx, cy, sx, sy))) < 1e-3
            cx, cy, sx, sy = new
            if done:
                break

    scale = 1.0 if pixel_size is None else float(pixel_size)
    return {
        "cx": cx * scale,
        "cy": cy * scale,
        "sigma_x": sx * scale,
        "sigma_y": sy * scale,
        "sigma_r": float(np.hypot(sx, sy)) * scale,
    }