# Traffic Light Vehicle Detection and Speed Estimation

This Python script uses YOLO (You Only Look Once) object detection model to detect, track, count, and estimate the speed of vehicles (cars, motorcycles, buses, and trucks) in a video feed. It processes a video file, defines a region of interest (ROI), and outputs vehicle counts and speeds to a CSV file while displaying real-time annotations on the video.

## Features

- **Vehicle Detection and Tracking**: Detects and tracks vehicles using YOLOv8 model with ByteTrack tracker.
- **Region of Interest (ROI)**: Defines a polygonal ROI to focus detection on specific areas.
- **Vehicle Counting**: Counts vehicles crossing a predefined line within the ROI.
- **Speed Estimation**: Calculates average speed of vehicles in km/h based on pixel movement and time.
- **Real-time Display**: Shows annotated video with bounding boxes, IDs, labels, and speeds.
- **CSV Output**: Saves vehicle speeds to `vehicle_speeds.csv`.
- **Supported Vehicles**: Car, Motorcycle, Bus, Truck.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO (`ultralytics`)
- A video file (e.g., MP4) for processing

## Installation

1. Install the required Python packages:
   ```
   pip install opencv-python numpy ultralytics
   ```

2. Download the YOLOv8 model weights (`yolov8n.pt`) if not already present. The script assumes it's in the same directory.

## Usage

1. Update the `VIDEO_PATH` variable in the script to point to your video file:
   ```python
   VIDEO_PATH = "path/to/your/video.mp4"
   ```

2. Run the script:
   ```
   python trafic_light.py
   ```

3. The script will process the video frame by frame:
   - Display the video with ROI, count line, detected vehicles, and speeds.
   - Show a separate window for the ROI detection area.
   - Press 'q' to quit.

4. After processing, check `vehicle_speeds.csv` for the speed data.

## Configuration

- **Video Resolution**: Set `WIDTH` and `HEIGHT` (default: 640x480).
- **ROI Points**: Adjust `bottom_left`, `bottom_right`, `top_right`, `top_left` to define the ROI polygon.
- **Count Line**: Modify `count_line_y` for the counting line position.
- **Pixel to Meter Ratio**: Update `PIXEL_TO_METER` for accurate speed calculation (default: 0.5).
- **Detection Confidence**: Change `conf` in `model.track()` (default: 0.4).
- **Tracker**: Uses "bytetrack.yaml"; ensure it's available.

## Output

- **Console**: Prints "End of video." when processing finishes.
- **CSV File**: `vehicle_speeds.csv` with columns: Vehicle ID, Speed (km/h).
- **Display**: Annotated video windows showing detections and counts.

## Notes

- Ensure the video file path is correct and accessible.
- The script assumes the YOLO model is pre-trained for COCO dataset classes.
- Speed estimation is approximate and depends on the `PIXEL_TO_METER` ratio.
- For better performance, adjust ROI and parameters based on your video.

## Troubleshooting

- If the video doesn't open, check the file path and format.
- Low detection accuracy: Increase confidence threshold or use a larger model (e.g., yolov8s.pt).
- Speed calculation issues: Calibrate `PIXEL_TO_METER` with known distances.

## License

This script is provided as-is for educational purposes.
