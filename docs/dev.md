
## Development Notes when running scripts and demos

General notes:
- The explanation for the respective demo's `config.yaml` can be found in the demo's code

### HDMI Demo explained

This script will run the **Gaze-v2** demo with sdk, showing the result on the attached monitor

1. Connect the raspberry pi to a HDMI monitor
2. Checkout/Download the Source Code from the `release` branch
3. `make install`
4. `.venv/bin/python scripts/show_result_to_hdmi.py --demo gaze-v2`
   1. You can edit the demo config at `./demos/gaze-v2/config.yaml` or use the web demo

Note:

- You can run other demos by specifying them with the `--demo` argument
  - For example you can run the `helmet` demo with `.venv/bin/python scripts/show_result_to_hdmi.py --demo helmet`

### Web UI Explained

This script will run the web ui allowing access to the sdk and demos over the network through a browser

1. Checkout/Download the Source Code from the `release` branch
   1. Ensure the `raspi-cam-srv-fork` submodule is pulled as well, if the folder is empty run
   2. `git submodule update --init --recursive`
2. `make install`
3. `.venv/bin/python scripts/show_result_to_web_ui.py`
4. Visit `http://localhost:9000` if you are on the raspberry pi itself or `http://RASPBERRYPI_IP:9000` if you are on the same network

Note:

- When running over wifi, take note of the resolution of the upload video stream
  - If the resolution is to high, this could cause degradation in the wifi network
    - For example if you run an upload resolution of `1920x1080`
  - We recommended starting from a lower resolution of around `640x480` and working your way up, monitoring the performance of the network as you do
- When running over lan, the upload resolution should not affect the network and it runs smoothly in real time (it could slightly affect the wifi network but not by much)
  - Just make sure both the rasberrypi and client machine are connected over lan and using their respective lan ip

## Developer Notes

1. Make a copy of the `demos/template` folder to `demos/DEMO_NAME`
   1. Look at the `demos/DEMO_NAME/demo_pipeline.py`, `DemoPipeline` class
      1. Override the methods `post_processing`, `draw_metadata` and `serialize_metadata`
         1. `post_processing` and `draw_metadata` for the `scripts/show_result_to_hdmi.py` script
         2. `post_processing` and `serialize_metadata` for the `scripts/show_result_to_hdmi.py` script
   2. You can also add your own config variables to the `demos/DEMO_NAME/config.yaml.template`
      1. (Optional) You can also update `AdditionalDemoConfig` in `demos/DEMO_NAME/demo_pipeline.py` for default values for the config variables
2. Create `demos/DEMO_NAME/config.yaml` based on `demos/DEMO_NAME/config.yaml.template`
3. You can run the scripts with `--demo` argument
   1. You can run the scripts with the argument `--demo DEMO_NAME` like `--demo gaze-v2`
4. Run the script with `.venv/bin/python scripts/SCRIPT_NAME` and it will run the sdk with your code
   1. I.e. `.venv/bin/python scripts/show_result_to_hdmi.py`

### Demo related code

- You can find all demo related code in the respective `demos/[DEMO_NAME]` folder
  - For example the `gaze-v2` demo is located in the `demos/gaze-v2` folder

We will be using `gaze-v2` as the example for the subsequent explanations
- The model files for the demo will be in the `demos/gaze-v2/models` folder
  - The `.rpk` file is the main model that runs on the imx500
  - The additional `.onnx` files are models we run as part of our post-processing pipeline
    - In this case we have another `mobilenetv4small.onnx` model we run for more accurate age prediction
- You can find all the code that runs as part of the demo pipeline in `demos/gaze-v2/demo_pipeline.py`


## Preping a release

The rationale behind this is that during development, we do not want to keep commiting changed configs for demos. As such in the development branches, none of the `config.yaml` file changes are tracked. However, for a release, since it has to be able to be downloaded and ran, it needs to include a working `config.yaml` file. Do also commit the model weight files in as well.

## More dev notes
- pi 10.68.254.118 is my main development pi
  - pi 5 8gb RAM
  - Staring at the screen
- pi 10.68.254.117
  - pi near the big tv
  - Used for 2nd dev and run live demo
- pi 10.68.254.125
  - pi connected to the monitor showing the live video

## Profiling

4k

- Camera FPS: 10
- Inference FPS: 5

2K

- Camera FPS: 30
- Inference FPS: 10.8

### 2K

- Post-Processing
  - This includes tensor calculations as well as image operations
  - Around 0.004s-0.01 (~4ms-10ms)
  - Comes to around 10.2 FPS
- Sending HTTP requests on LAN to another raspberry pi
  - Bottleneck is on Network and processing
  - Actual Front-End FPS: ~2.3
- On Wifi to dev laptop
  - Main bottleneck is network
  - Actual Front-End FPS: 2-5
- On Wifi to dev laptop (Just metadata)
  - Still bottlenecked on the network and processing
  - Actual Front-End FPS: 2-5
- Running DemoHub and SDK on the same raspberry pi

  - Actual Front-End FPS: ~8

- Running SORT + network.rpk

  - 10 FPS on camera, same results everywhere else

- Running Mobilenetv4Small
  - 0.015-0.02s per inference (Depending on cv2 large face image size)
  - Even just 3 faces => Extra ~0.045s => 0.1s to 0.145s per frame
    - => 6.8 FPS
  - Running mobilenet slows down the whole pipeline

## Error Encountered

- Sometimes the sdk just stops working and hangs
  - No Known solution other than increase buffer counts and timeouts?
  - Could also detect when the process is not updating and restart it
  - https://github.com/raspberrypi/picamera2/issues/1090
  - Somehow, setting a smaller `buffer_count` size of `2` makes it quite stable

Possible reason is that during `img = request.make_array("main")`, since the img array is still being referenced, the request is not able to be released.
Adding a `.copy()` to the img array seems to fix this?
