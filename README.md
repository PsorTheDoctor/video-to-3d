# Video to 3D
Reconstruct a geometrically coherent 3D scene from a video clip!

![Demo](results/demo.gif)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/PsorTheDoctor/video-to-3d.git
cd video-to-3d
```
2. Ensure Python interpreter is installed. Then create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Execute the program on the example video clip:
```bash
python main.py
```
Or run it on your own videos (taken from the `data` folder by default):
```bash
python main.py --video_path your_video.mp4
```

### Command-line arguments
* `--video_path`: Path to the input video file.
* `--save`: Save the final point cloud to the `results` folder.
