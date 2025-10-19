# 🖌️ Air Draw: Ghost Draw

This project is an **interactive air-drawing game** using **OpenCV**, **TensorFlow**, and **Pygame**, where you can draw symbols in the air with your hand to defeat incoming ghosts.  

It uses **hand tracking**, **gesture recognition**, and a trained deep learning model to detect drawn shapes in real time.  

## ✨ Features
- 🎮 **Hand gesture control** using webcam  
- 🧠 **Deep learning model** to recognize drawn shapes (`down`, `horz`, `up`, `vert`)  
- 👻 Ghost enemies that approach the player  
- ❤️ Lives and score system  
- 🪄 Magical wand overlay for immersive experience  
- 💥 Real-time blending of transparent images on video feed

---

## 🧰 Tech Stack

- [Python 3.x](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Pygame](https://www.pygame.org/)
- [Flask](https://flask.palletsprojects.com/)
- Custom Hand Tracking Module (`HandTrackingModule.py`)
- Custom Ghost Class (`Ghost.py`)

---

## 🧠 How It Works

1. The webcam captures your hand movements in real time.
2. The **hand tracking module** detects hand position and orientation:
   - ✍️ Vertical fist → Drawing mode
   - ✨ Horizontal fist → Prediction mode
3. When in prediction mode, the model classifies the drawn shape.
4. If the drawn shape matches the ghost’s symbol, the ghost is defeated.
5. If a ghost collides with the heart, you lose a life.
6. Score increases with each defeated ghost; difficulty scales with score.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/huskiehackers/OpenSourceHackathonAirGhosts.git
cd OpenSourceHackathonAirGhosts
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run App
```bash
python app.py
or
python3 app.py
```

## 📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute with attribution.

