"""
camera.py - Helper functions for The Imaging Source DMK 21AF618 using the
official `imagingcontrol4` (IC4) Python package.

Requirements
------------
    pip install imagingcontrol4 numpy

IC4 talks to cameras through GenTL drivers. The DMK 21AF618 is a FireWire
(IEEE 1394) camera, so you also need The Imaging Source "IC4 GenTL Driver for
Legacy Devices" (plus the FireWire device driver) installed. The old DirectShow
driver alone (what IC Capture uses) is NOT enough for IC4.

Typical use
-----------
    import camera

    cam = camera.init_camera(exposure_s=0.01, bitdepth=8)
    img = camera.take_picture(cam)          # 2D numpy array
    camera.set_exposure(cam, 0.005)
    camera.set_bitdepth(cam, 8)
    camera.close_camera(cam)
"""

import warnings
from dataclasses import dataclass

import numpy as np
import imagingcontrol4 as ic4

_BITDEPTH_TO_PIXELFORMAT = {8: "Mono8", 16: "Mono16"}

_library_users = 0  # reference count so Library.init/exit are called once


@dataclass
class Camera:
    """Handle returned by init_camera()."""
    grabber: ic4.Grabber
    sink: ic4.SnapSink
    bitdepth: int = 8


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _library_acquire():
    global _library_users
    if _library_users == 0:
        ic4.Library.init()
    _library_users += 1


def _library_release():
    global _library_users
    _library_users = max(0, _library_users - 1)
    if _library_users == 0:
        ic4.Library.exit()


def _start_stream(cam):
    cam.grabber.stream_setup(
        cam.sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START
    )


def _stop_stream(cam):
    if cam.grabber.is_streaming:
        cam.grabber.stream_stop()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def list_cameras():
    """Return a list of (index, model_name, serial) for all cameras found."""
    _library_acquire()
    try:
        return [(i, d.model_name, d.serial) for i, d in enumerate(ic4.DeviceEnum.devices())]
    finally:
        _library_release()


def init_camera(index=0, serial=None, bitdepth=8, exposure_s=None):
    """Open the camera, configure it and start streaming.

    Parameters
    ----------
    index : int        Position in the device list (used if `serial` is None).
    serial : str       Select a camera by serial number instead of index.
    bitdepth : int     8 (Mono8) or 16 (Mono16, if supported by the camera).
    exposure_s : float Exposure time in seconds. None = leave as is.

    Returns
    -------
    Camera handle to pass to the other functions.
    """
    _library_acquire()
    try:
        devices = ic4.DeviceEnum.devices()
        if not devices:
            raise RuntimeError(
                "No camera found. Check the cable/power and that the IC4 GenTL "
                "driver for legacy (FireWire) devices is installed."
            )

        if serial is not None:
            matches = [d for d in devices if d.serial == serial]
            if not matches:
                raise RuntimeError(f"No camera with serial {serial!r} found.")
            dev = matches[0]
        else:
            dev = devices[index]

        grabber = ic4.Grabber()
        grabber.device_open(dev)
        cam = Camera(grabber=grabber, sink=ic4.SnapSink())

        set_bitdepth(cam, bitdepth)
        if exposure_s is not None:
            set_exposure(cam, exposure_s)

        _start_stream(cam)
        return cam
    except Exception:
        _library_release()
        raise


def take_picture(cam, timeout_ms=2000, n_average=1):
    """Grab one image and return it as a 2D numpy array.

    Parameters
    ----------
    timeout_ms : int   Max time to wait for a frame.
    n_average : int    If > 1, average that many frames (returns float64).
    """
    frames = []
    for _ in range(max(1, n_average)):
        buf = cam.sink.snap_single(timeout_ms)
        frames.append(np.squeeze(buf.numpy_copy()))  # (H, W, 1) -> (H, W)

    if len(frames) == 1:
        return frames[0]
    return np.mean(np.stack(frames).astype(np.float64), axis=0)


def set_exposure(cam, exposure_s):
    """Set the exposure time in seconds (e.g. 0.01 = 10 ms).

    Turns off auto-exposure, converts to microseconds (IC4 unit) and clamps to
    the range supported by the camera. Returns the exposure actually set, in seconds.
    """
    if exposure_s <= 0:
        raise ValueError("exposure_s must be > 0 (seconds).")

    pmap = cam.grabber.device_property_map
    try:
        pmap.set_value(ic4.PropId.EXPOSURE_AUTO, False)
    except ic4.IC4Exception:
        pass  # some devices/drivers have no auto-exposure switch

    prop = pmap.find_float(ic4.PropId.EXPOSURE_TIME)
    value_us = float(np.clip(exposure_s * 1e6, prop.minimum, prop.maximum))
    pmap.set_value(ic4.PropId.EXPOSURE_TIME, value_us)

    return pmap.get_value_float(ic4.PropId.EXPOSURE_TIME) * 1e-6


def set_bitdepth(cam, bitdepth):
    """Select 8 or 16 bit output (Mono8 / Mono16 pixel format).

    The stream is stopped while the format changes, then restarted. Raises
    RuntimeError if the camera does not offer the requested format.
    """
    if bitdepth not in _BITDEPTH_TO_PIXELFORMAT:
        raise ValueError("bitdepth must be 8 or 16.")

    name = _BITDEPTH_TO_PIXELFORMAT[bitdepth]
    pmap = cam.grabber.device_property_map

    was_streaming = cam.grabber.is_streaming
    _stop_stream(cam)
    try:
        try:
            pmap.set_value(ic4.PropId.PIXEL_FORMAT, name)
        except ic4.IC4Exception:
            available = [e.name for e in pmap.find_enumeration(ic4.PropId.PIXEL_FORMAT).entries]
            raise RuntimeError(
                f"Pixel format {name} not supported by this camera. Available: {available}"
            )
        cam.bitdepth = bitdepth
    finally:
        if was_streaming:
            _start_stream(cam)


def close_camera(cam):
    """Stop streaming, close the device and release the IC4 library."""
    if cam is None:
        return
    try:
        _stop_stream(cam)
        if cam.grabber.is_device_open:
            cam.grabber.device_close()
    finally:
        _library_release()


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


if __name__ == "__main__":
    print("Cameras found:", list_cameras())
    cam = init_camera(exposure_s=0.01, bitdepth=8)
    img = take_picture(cam)
    print("Image:", img.shape, img.dtype, "min/max:", img.min(), img.max())
    print("Beam moments:", beam_std(img))
    close_camera(cam)