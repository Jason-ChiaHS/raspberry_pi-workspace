# Raspberry Pi IMX500 SDK

SDK is mainly for:

- Testing models performance locally using the cv2 preview
- Running it as a ImageMeta server
- Running the entire webui demo

## Running the Demo

To clone the project, run:
```
git clone GIT_REPO_URL
git submodule update --init --recursive
```

You need to run this to also clone the submodules used in the project. If not the submodules folder will be empty.

### 1. Pulling Source Code and running just the demo with HDMI

This is faster and requires lesser dependencies installed on the raspberry pi.

1. `make install-hdmi`
2. `make run-hdmi-gaze-v2` - To run the HDMI, gaze-v2 demo

### 2. Running the WebUI using the release build with the release image

You can find the release build image from the [pentas vision sharepoint](https://pentasvision.sharepoint.com/sites/pentas_imx500_sdk/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2Fpentas%5Fimx500%5Fsdk%2FShared%20Documents%2F30%2E%20Pentas%20Releases%2FDemohub%2FSDcard%5FImage&viewid=89eb4217%2D5222%2D4516%2D974b%2D01811911d82f).

1. `make install`
2. `make run-webui`

### 3. Running manually

Prerequisites
- IMX500 libraries
  - https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Picamera2
  - https://pypi.org/project/picamera2/
  - https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
  - `sudo apt install python3-picamera2 --no-install-recommends`
- opencv
  - `sudo apt-get install python-opencv`
  - apt install is preferred to using python
- git clone
  - Clone down the submodules
  - `git submodules update --init --recursive`
- python
  - create venv
    - `python -m venv .venv --system-site-packages`
  - sdk libs
    - `./.venv/bin/pip install -r requirements.txt`
    - `./.venv/bin/pip install -e .`
  - raspi-cam-srv libs
		- `./.venv/bin/pip install -r raspi-cam-srv-fork/requirements.txt; \`
		- `./.venv/bin/pip install -e raspi-cam-srv-fork; \`
- node
  - install webui libs
    - `cd scripts/web_ui`
    - `npm i`

### 4. Alternatives
- [Aitrios RPi sample apps](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps)
