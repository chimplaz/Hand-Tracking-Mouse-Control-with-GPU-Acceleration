
Hand Tracking Mouse Control with GPU Acceleration
This project leverages computer vision and GPU-accelerated processing to create a hand-tracking-based mouse control system. The system uses OpenCV, Mediapipe, and CuPy to detect hand landmarks in real time and translates hand movements into mouse cursor actions. The goal is to control the mouse pointer and perform actions like clicking and scrolling using hand gestures.
SYSTEM REQUIREMENTS
Hardware Requirements:
Webcam: A functional webcam for capturing hand gestures (built-in or external).
Computer with GPU: (A CUDA-compatible GPU (e.g., NVIDIA GPU) for accelerating computations with CuPy)

    NVIDIA GPUs (CUDA-Compatible): 
Consumer GPUs (GeForce Series):
Desktop GPU:
NVIDIA GeForce GTX 900 Series and newer (e.g., GTX 960, GTX 1050, GTX 1660, RTX 2060, RTX 3060, RTX 4060).
Laptop GPUs:
NVIDIA GeForce MX Series (e.g., MX250, MX450; entry-level performance).
NVIDIA GeForce GTX/RTX Mobile GPUs (e.g., GTX 1650, RTX 3050).
Sufficient VRAM (Video Memory):
•	2GB minimum for basic performance.
•	4GB or more is recommended for smooth operation, especially with high-resolution webcams or large screen resolutions.
Minimum System Specs:
•	Processor: Intel Core i5 or AMD Ryzen 5 or equivalent.
•	RAM: At least 4GB (8GB or more recommended).
•	Operating System: Windows or Linux.

Software Requirements:
Python 3.7 or newer.
Python Libraries:
Install the following libraries using pip:
•	opencv-python: For capturing and processing video frames.
•	mediapipe: For hand-tracking functionality.
•	pyautogui: For controlling the mouse and performing clicks.
•	cupy: For GPU acceleration (requires NVIDIA CUDA drivers).
•	numpy: For numerical calculations.
Command to install: pip install opencv-python mediapipe pyautogui cupy numpy

NVIDIA CUDA Toolkit:
Required for using CuPy with NVIDIA GPUs. Install the appropriate version compatible with your GPU and operating system.

Visual Studio C++
You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options.
IDE/Text Editor (Optional):
•	Recommended: PyCharm, VSCode, or Jupyter Notebook for code development and debugging.

