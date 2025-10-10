# 2026 Radar
## Timeline
### Single Camera
- Create a more easy to read version of last year's CV code
- Start multi-layer object detection
	- Create and implement a YOLO model for detecting robots (use pre trained model for detecting cars as substitute)
	- Create and implement a YOLO model for detecting armour panels
		- Run it on a cropped image of just the robot (much smaller model)
	- Create and implement a RESNET model for image classification to determine type of armour panel
		- Run on a cropped image of just the armour plate for better object classification
	- Make the layers work with each other

### Stereo Camera
- Design a mount for the two cameras
- Figure out a method for calibrating the two cameras
- Use the cropped image from one camera and run keypoint detection
- Use size of bounding box to get rough estimate of distance and create a mask for the second camera
- Match keypoints from first and second image to get location of robot on second image
- Perform distance calculations

### Mapping on Playing Field
- Get a model/map of the playing field
- Calculate location of robot based on direction of detection and distance
- Figure out how to send data to game server


