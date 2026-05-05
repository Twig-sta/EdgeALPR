# Raspberry Pi Zero 2 W Setup

This project is designed to run on Raspberry Pi OS with a Raspberry Pi Camera.

## System Packages

Install the camera, OCR, and OpenCV runtime packages:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv tesseract-ocr
```

## Python Packages

If you are using a virtual environment:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

Using `--system-site-packages` lets the virtual environment see Raspberry Pi OS packages like `picamera2`.

## Camera Settings

The app defaults to `640x480`, which is a safer live-stream size for the Pi Zero 2 W.

Optional overrides:

```bash
export EDGEALPR_CAMERA_WIDTH=640
export EDGEALPR_CAMERA_HEIGHT=480
export EDGEALPR_STREAM_QUALITY=70
export EDGEALPR_STREAM_DELAY=0.08
```

Lower quality or higher delay will reduce CPU load. For example, `EDGEALPR_STREAM_DELAY=0.12` streams at about 8 FPS.

## Run

```bash
source venv/bin/activate
python dashboard/app.py
```

Then open:

```text
http://<raspberry-pi-ip>:5055
```
