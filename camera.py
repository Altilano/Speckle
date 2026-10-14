"""
camera.py - Helper functions for The Imaging Source DMK 27AUR0135 (monochrome,
USB 3 camera of the "27U" series) using the official `imagingcontrol4` (IC4) package.

Requirements
------------
    pip install imagingcontrol4 numpy

IMPORTANT - drivers (Windows): `pip install imagingcontrol4` installs only the Python
bindings. IC4 reaches cameras through GenTL producers, installed separately from
https://www.theimagingsource.com/en-us/support/download/ . The 27U series is a
"Non-GenICam" (legacy) camera, NOT USB3 Vision, so you need:
  1. the download called "GenTL Producer for Non-GenICam Cameras (DirectShow)"
     (installs the "IC4 GenTL Driver for Legacy Devices"), and
  2. the matching DirectShow device driver ("Device Driver for USB Cameras", the one
     IC Capture uses).
The "IC4 GenTL Driver for USB3Vision Devices" (32U/33U/37U/38U) will load but will not
list this camera. After installing, fully restart VS Code/the terminal so the updated
GENICAM_GENTL64_PATH is picked up, then run this file: if no camera is found it calls
diagnose(), which shows what IC4 sees.

Typical use
-----------
    import camera

    cam = camera.init_camera(exposure_s=0.01, bitdepth=8)
    img = camera.take_picture(cam)                      # 2D numpy array
    camera.set_exposure(cam, 0.005)
    camera.set_bitdepth(cam, 16)
    stats = camera.beam_std(img, pixel_size=camera.PIXEL_SIZE)
    camera.close_camera(cam)
"""

import gc
from dataclasses import dataclass

import numpy as np
import imagingcontrol4 as ic4

# Pixel formats requested for each bit depth. Which ones the camera really offers depends
# on the driver mode (legacy/DirectShow cameras often only provide Mono8 = "Y800");
# set_bitdepth() raises an error listing the available formats if one is missing.
_BITDEPTH_TO_PIXELFORMAT = {8: "Mono8", 16: "Mono16"}

# Pixel pitch in meters (3.75 um, assuming the onsemi AR0135 sensor inferred from the
# model name; confirm with the camera data sheet). Use as beam_std(..., pixel_size=PIXEL_SIZE).
PIXEL_SIZE = 3.75e-6

_library_ready = False  # IC4 is initialized once and kept alive (see shutdown_library)


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
    """Initialize IC4 on first use. It is deliberately NOT shut down after every call:
    IC4 objects (Grabber, Sink, ...) must not outlive Library.exit(), otherwise their
    destructors print 'Library.init was not called' when Python exits."""
    global _library_ready
    if not _library_ready:
        ic4.Library.init()
        _library_ready = True


def _library_release():
    """Kept as a hook; the library stays initialized (see shutdown_library)."""


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


def diagnose():
    """Print what IC4 can see, to find out why the camera is not listed.

    Shows the Python bitness, the GenTL search paths (and the .cti driver files found
    in them), IC4's own log messages and every interface/device it detects.
    """
    import glob
    import os
    import platform
    import struct

    print(f"Python {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
    print("imagingcontrol4:", getattr(ic4, "__version__", "unknown version"))

    for var in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        val = os.environ.get(var)
        print(f"{var} = {val!r}")
        for folder in (val or "").split(os.pathsep):
            if folder:
                files = [os.path.basename(f) for f in glob.glob(os.path.join(folder, "*.cti"))]
                print(f"    {folder}: {files or 'NO .cti files'}")

    paths = os.environ.get("GENICAM_GENTL64_PATH", "")
    if not any(k in paths.lower() for k in ("legacy", "directshow", "non-genicam")):
        print("HINT: no legacy/non-GenICam producer in GENICAM_GENTL64_PATH. The 27U series "
              "needs 'GenTL Producer for Non-GenICam Cameras (DirectShow)' (the USB3Vision "
              "producer does not list this camera). Install it and restart VS Code/the terminal.")

    global _library_ready
    if not _library_ready:
        try:  # with IC4 log output, which says which producers were loaded or rejected
            ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)
        except (TypeError, AttributeError):
            ic4.Library.init()
        _library_ready = True
    try:
        interfaces = list(ic4.DeviceEnum.interfaces())
        print(f"Interfaces found by IC4: {len(interfaces)}")
        for itf in interfaces:
            print("  -", getattr(itf, "display_name", itf),
                  "| producer:", getattr(itf, "transport_layer_name", "?"))
            for d in getattr(itf, "devices", []):
                print("       device:", d.model_name, d.serial)
        print("Devices found by IC4:", [(d.model_name, d.serial) for d in ic4.DeviceEnum.devices()])
    finally:
        interfaces = itf = d = None  # release IC4 objects before the library shuts down
        _library_release()


def init_camera(index=0, serial=None, bitdepth=8, exposure_s=None):
    """Open the camera, configure it and start streaming.

    Parameters
    ----------
    index : int        Position in the device list (used if `serial` is None).
    serial : str       Select a camera by serial number instead of index.
    bitdepth : int     8 (Mono8) or 16 (Mono16).
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
                "No camera found by IC4. Check the USB 3 cable/port and that the camera is "
                "not open in another program (e.g. IC Capture). The DMK 27AUR0135 is a "
                "legacy (non-GenICam) camera: install 'GenTL Producer for Non-GenICam "
                "Cameras (DirectShow)' plus the matching DirectShow driver, restart VS "
                "Code, and run diagnose() to see what IC4 detects."
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
    """Stop streaming and close the device (IC4 itself stays initialized for reuse)."""
    if cam is None:
        return
    try:
        _stop_stream(cam)
        if cam.grabber.is_device_open:
            cam.grabber.device_close()
    finally:
        cam.sink = None
        cam.grabber = None
        _library_release()


def shutdown_library():
    """Optional: shut IC4 down completely (normally not needed, the OS cleans up on exit).

    Call it only after every camera was closed with close_camera() AND every handle
    was deleted (`del cam`), otherwise IC4 prints destructor errors."""
    global _library_ready
    if _library_ready:
        gc.collect()
        ic4.Library.exit()
        _library_ready = False




if __name__ == "__main__":
    cameras = list_cameras()
    print("Cameras found:", cameras)
    if not cameras:
        diagnose()
        raise SystemExit(1)
    cam = init_camera(exposure_s=1/10, bitdepth=8)
    img = take_picture(cam)
    print("Image:", img.shape, img.dtype, "min/max:", img.min(), img.max())
    close_camera(cam)