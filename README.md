# PoEAI
## A Deep-Learning Based AI for Path of Exile

See [my blog series](https://nicholastsmith.wordpress.com/2017/07/08/a-deep-learning-based-ai-for-path-of-exile-a-series/) for more details on the individual components.

### Bot.py

The class that contains the main bot loop.

### BotDebugger.py

A class to help with debugging the main program.

### Main.py

Program entry point

### MovementMap.py

Class that keeps track of the bot's internal representation of the world. Contains a dictionary which maps 3D positions like (x,y,z) to a label (open, obstacle, item, etc).

### ProjMap.py

Handles converting from 3D to 2D coordinates and visa-versa based on a projection matrix calibrated for Path of Exile.

### ScreenViewer.py

Code to grab image data from the screen using Windows API

### TargetingSystem.py

Class for classifying image data from the game. Used to identify obstacles, enemies, items, and lightning warp (for movement).

### TFModel

Contains pre-trained tensorflow models

### FAQ

**What is "CNNC" and why can't I import it?**

CNNC stands for Convolutional Neural Network Classifier. It is from a neural network library I wrote here: https://github.com/nicholastoddsmith/pythonml **Note**: Currently, you will need to clone an older version where CNNC still exists or refactor TargetingSystem to use the newer API *ANNC*. (If you do this, *please* make a pull request).

**Can I use this to farm for currency/experience?**
**No.** The code is currently proof-of-concept quality and by no means production ready. The code is made available primarily for other researchers and *by no means intended as an end-user application*.
