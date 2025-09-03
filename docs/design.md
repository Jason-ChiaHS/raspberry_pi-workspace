# Design Doc

This doc provides the high level overview and what you need to know for this project.
This project is the raspberry pi IMX500 SDK. Very broadly, it has 2 parts:

1. A python Software Development Kit(SDK) which allows developers to run their `.rpk` models on the raspberry pi and implement whatever they need in python with the output tensors of the model. This runs per frame on the IMX500, going as fast as the model can run on the IMX500 (~10fps).
2. For sales/business people to run demos we have already made as part of the project

## Background Knowledge

This will briefly go over the background knowledge that is good to know if you are working on the SDK.
There is another section [here]() if you are looking for the background knowledge needed to start writing a demo and running your `.rpk` model on the IMX500.

### Python Libraries

We make use of the following python libraries quite extensively in the SDK.

#### PiCamera2

This is the library we use to control anything related to the Camera. Notable sections include:

- Initalizing the Camera
- How the [IMX500 is integrated](https://github.com/raspberrypi/picamera2/tree/main/picamera2/devices/imx500) into PiCamera2
- How to drive the event loop for the camera
- Multi-Threading and passing control back to the camera's event loop
- The opencv RGB format for frames from the camera
- Configuring the camera parameters
- Camera Controls

You can also refer to their [examples](https://github.com/raspberrypi/picamera2/tree/main/examples) to see how PiCamera2 code looks likes.

References:

- [PiCamera2 Github](https://github.com/raspberrypi/picamera2)
- [PiCamera2 Manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

#### OpenCV

We use this library for anything realted to image manipulation.

- Convert between different color formats
- Resizing images
- Encoding images to strings


#### Numpy

We use this for anything related to tensor manipulation

#### Onnxruntime

This is used to run other models on the cpu. It only requires a `.onnx` file and can run the same model on different hardware. This makes it easy to test the model on a PC, before running the model on the raspberry pi.

References:
- [Docs](https://onnxruntime.ai/docs/)
- [Github](https://github.com/microsoft/onnxruntime)

#### SocketIO

This is included as this library is extensively used to send data between the front-end web UI and back-end demo. You should be familiar with this if you intend to work on the WebUI scripts below.

#### Svelte

This was the front-end library that was used to create the front-end. There was no particular reason for picking other than it was familiar with the framework and it has open source libraries for many of the components needed.

References:
- https://svelte.dev/

### Raspbian Operating System(OS)

This section includes topics to setup the OS to run the IMX500 as well as tools to help develop the SDK.

#### System Packages

You can find the relevant details for setting up the IMX500 on the Raspberry Pi [here](https://www.raspberrypi.com/documentation/accessories/ai-camera.html). You can also find how to clone and prepare a development image [here](https://pentas-gitlab.southeastasia.cloudapp.azure.com/pentas-member/raspberry-pi-imx500-imager)

#### System Tools

Just pointing out a few tools

- Looking at `Cma` from `/proc/meminfo` can let you know if you are running out of memory to allocate to the camera
  - `./development/scripts/mem-check.sh`
- There might be some hardware acceleration quirks when running cv2 demos.
- You can use the built-in `vncserver` that the raspberry pi has and a corresponding `vncviewer`
- Some sort of editor or IDE connected to the raspberry pi
  - This is to allow for easier development as you need to run the SDK on the raspberry pi
  - I personally used VsCode with the `remote: ssh` extension to connect to the raspberry pi

### Converting Models to the `.rpk` format

There are 2 steps to convert a normal `pytorch` or `tensorflow` model into a `.rpk` model. The first is to use the `imx500-converter` tools followed by the `imx500-package` tool.

References:
- [imx500-converter](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter?version=3.16.1&progLang=)
- [imx500-package](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-packager?version=2025-07-09&progLang=)

### Hardware Limitations

There are also a few hardware limitation to take note of:
- Power output for the pi
  - Try and use power outputs that meet the official raspberry pi's wattage
  - If not the camera might have enough power to run
- The camera could also not be detectable at the time
  - Do physically check the camera and ribbon cable connection
- Flashing new IMX500 firmware
  - Replace the `/lib/firmware/imx500_firmware.fpk` file and reboot the raspberry pi for the change to take place

## SDK

This section will briefly go over the important parts of the SDK. Generally, the flow is to run a script with a given demo. The script will then load the Demo (along with its config) and then calls the SDK to run the logic for the script.

### cam.py

The abstraction over the PiCamera2 library. More details in the [architecture diagram](../arch.drawio).

### base_pipeline.py

Contains the base class extended by the Demos and template. Will be automatically called by the scripts when it runs with the demo.

### Config

The SDK contains helpers to load the demo `config.yaml` files and parse the values.

### Templates and Demo

You can find the relavant code at `./demos`. Every demo is built based on the `./demos/template` folder. You should treat the template folder as a normal python module with the entrypoint being the `DemoPipeline` class in the `demo_pipeline.py` file. All demo related logic should be placed in the class following normal python development best practices, allowing for any imports in the demo folder itself.

### Scripts

The scripts can be found in the `./scripts` folder and contain the code that calls the corresponding demo as well as the SDK to get the behaviour we want. For example, the `show_result_to_hdmi.py` script will start the demo and show(write to screen using opencv) the image capture from the camera along with the drawn metadata. (Implemented by the demo)

### WebUI Script

This gets it own section as it is vastly more complicated compared to the other scripts. This script run 2 flask servers, one being the `raspi-cam-srv-fork` which allows for the user to set camera controls on the IMX500 while the over server controls the demo running on the raspberry pi. This script can only allow 1 user to control the camera at any one time but can allow for multiple camera feeds at the same time. It is recommended that you study the source code for this directly.

### raspi-cam-srv-fork

A fork of an open source project as we wanted to integrate its already built front-end camera control capabilities into the demo. You can find the original source code [here](https://github.com/signag/raspi-cam-srv).

### Development Folder

The python scripts in this folder are mainly reference code or simple scripts to test certain functionalities. Feel free to change and use them as you see fit.

## GazeV3 Demo

You can refer to the [architecture diagram](../arch.drawio) for a more detailed breakdown.

## Helmet Demo

For the helmet demo, the output tensor per frame are processed to get the bounding boxes, class and confidence scores. This is then drawn on the image and passed back to the script.

## Good numbers and performance metrics to know

Depending on the size of the model (number of parameters/layers/ops) the fps of just running the model alone could vary. I have seen the fps range anywhere from 7-17fps so far. However, the fps of running the same model should be consistent.

As such I recommend checking the "native" fps of whatever model you run to know if anything in the pipeline is affecting the real-time performance.

Running this SDKs and demo on a raspberry pi4/5 with 4GB of RAM is possible. However, you should not run any other program on the raspberry pi, as there is a good chance you might run into an issue where the camera has reserved all the remaining RAM causing a freeze. The only way to fix this as far as I know is to reset the power to the raspberry pi.

## Useful Links

- [Aitrios Development Website](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera)
  - Look for the pages related to the IMX500
