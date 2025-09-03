# Current Issues
- Still crashes and freezes after running for a long time
  - But only when other processes are running on the pi (I.e. code development)
- We can only really use a buffer size of 2, for the camera and get stable fps
  - However the recommended is 10, but I have not tested any performance difference when it runs

# Future Improvements
- Add a config to allow db tracks to be saved, but not the face images
  - Just need to have the sqlite, if running to save empty strings or None for the face_img_path
- Touch up front-end like the fps counter etc.

## POC for multiple pis to use the same db

- script to run custom code for the demo
- There is a `http_central_server.py` file
- The raspberry pis will all send and use a central database with changes in config for the db string
