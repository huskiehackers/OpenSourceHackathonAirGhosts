# AirDraw: Virtual Drawing with Hand Gestures
## Overview
AirDraw is an innovative computer vision application that enables users to draw in the air using hand gestures. By tracking finger movements through a standard webcam, this system translates gestures into digital drawings on screen. With three distinct detection modes, color selection, and eraser functionality, AirDraw provides an intuitive interface for both creative expression and practical applications.

## Key Features
- **Gesture-based drawing:** Draw with single finger, "lift pen" with two fingers

- **Color selection:** Choose from red, blue, green, yellow pens

- **Three detection modes:**
  - Free drawing (no recognition)
  - Alphabet recognition
  - Number recognition
 
- **Canvas management:** Erase drawings or clear entire canvas

- **Machine learning integration:** Pre-trained CNN models for character recognition

- **Cross-platform interface:** Combines OpenCV and Pygame for optimal performance

## Requirements
- Python 3.7+

- Webcam (720p or higher recommended)

- Required model file:
    - ```bModel.h5``` (alphabet recognition)
    - ```bestmodel.h5``` (number recognition)

- Required header images (in ```header``` directory)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/air-draw.git
cd air-draw

# Install dependencies
pip install -r requirements.txt

# Verify installation
python Draw/app.py
```

## Alternative Setup
1. Create virtual environment:
```bash
python -m venv airdraw-env
source airdraw-env/bin/activate  # Linux/Mac)
airdraw-env\Scripts\activate    # Windows
```

2. Manually install dependencies:
```bash
pip install opencv-python==4.5.5.64 keyboard==0.13.5 mediapipe==0.8.9.1 numpy==1.19.5 pygame==2.0.1 tensorflow==2.5.0
```

## How to Use
1. Launch application: python Draw/app.py

2. Position hand in camera view (see gesture guide below)

3. Use control strip at top of screen:
   - Color selection: Move two fingers over color blocks
   - Eraser: Select eraser icon
   - Clear canvas: Select trash can icon
   - Exit: Select X icon
  
### Gesture Guide
| Action         | Gesture                           |
| -------------- | --------------------------------- |
| Draw           | Single index finger extended      |
| Move/Select    | Index + middle fingers extended   |	
| Lift "pen"     | Two fingers together              |

### Mode Controls
- Press ```O```: Free drawing mode (no recognition)
  
- Press ```A```: Alphabet recognition mode

- Press ```N```: Number recognition mode

## File Structure
```
air-draw/
├── Draw/
│   ├── app.py                  # Main application entry point
│   ├── HandTrackingModule.py   # Hand detection logic
│   └── VirtualPainter.py       # Core drawing functionality
├── header/                     # UI element images
│   ├── 1.jpg                   # Red pen icon
│   ├── 2.jpg                   # Blue pen icon
│   ├── 3.jpg                   # Green pen icon
│   ├── 4.jpg                   # Yellow pen icon
│   ├── 5.jpg                   # Eraser icon
├── models/
│   ├── bModel.h5               # Alphabet recognition model
│   └── bestmodel.h5            # Number recognition model
├── requirements.txt            # Dependencies list
└── README.md                   # This documentation
```

## Troubleshooting

### Issue: "Module not found" errors

Solution:
  1. Verify virtual environment activation
  2. Reinstall requirements: ```pip install -r requirements.txt```
  3. Check Python version (requires 3.7+)

### Issue: Camera not detected

Solution:
  1. Check camera permissions
  2. Try different camera index:
     ```py
     # In VirtualPainter.py, change:
     cap = cv2.VideoCapture(0)  # → cap = cv2.VideoCapture(1)
     ```

### Issue: Poor hand tracking

Solution:
  1. Ensure adequate lighting
  2. Maintain 1-2 meter distance from camera
  3. Use solid background contrast
  4. Clean camera lens

### Issue: Model loading errors

Solution:
  1. Verify model files are in ```models/``` directory
  2. Check TensorFlow version compatibility
  3. Ensure file paths are correct in code

## Customization Options

Modify these constants in ```VirtualPainter.py```:
```py
# Drawing parameters
brushThickness = 15          # Line thickness
eraserThickness = 30         # Eraser size
BOUNDRYINC = 5               # Recognition boundary padding

# Display settings
width, height = 1280, 720    # Screen resolution
```

## Recognition Models
| Model               | Type        | Classes      |
| ------------------- | ----------- | ------------ |
| ```bModel.h5```     | Alphabet    | 26 letters   |
| ```bestmodel.h5```  | Digits      |	0-9          |

Note: Models trained on EMNIST dataset with custom data augmentation

## Dependencies
| Package         | Purpose                           |
| -------------- | --------------------------------- |
| OpenCV           | 	Computer vision operations      |
| MediaPipe    | Hand tracking  |	
| TensorFlow     | Character recognition              |
| Pygame     | Canvas rendering              |
| Keyboard     | Mode switching              |

## Future Enhancements

- Gesture-based zoom/pan functionality

- Save/load drawing functionality

- Custom model training interface

- Multi-hand tracking support

- Export to image/PDF functionality

- Pressure sensitivity simulation

- 3D drawing space integration

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository

2. Create a feature branch (```git checkout -b feature/your-feature```)

3. Commit your changes (```git commit -am 'Add some feature'```)

4. Push to the branch (```git push origin feature/your-feature```)

5. Open a pull request

### Priority Areas for Contribution
- Performance optimization

- Additional recognition models

- Cross-platform compatibility improvements

- UI/UX enhancements

- Documentation translations

## Academic Use

This project is ideal for computer vision and HCI courses. Key learning areas include:

- Real-time object tracking

- Gesture recognition systems

- CNN model integration

- Human-computer interaction design

- Computer graphics rendering
