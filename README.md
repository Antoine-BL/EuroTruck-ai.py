# EuroTruck-ai.py

## Description
A WIP Self-driving AI for EuroTruck Simulator 2. I started this project for my end of studies project in Computer Science Technology degree at Collège Gérald-Godin

## Concept
The concept for this project was to use a Deep Convolutional Neural Network that could take images as input and output steering commands.
The primary goal was to create a model that could steer a truck in Euro Truck Simulator 2 and stay in a lane on a highway. 
This goal has not yet been achieved but I still intend to continue on this project.

## Challenges
To get the project to its current state, I had to overcome the following challenges:
- Make a program to register user input and frames from the game to gather data for the ai.
- Make a script that could take the output from the model and control the game.
- Write code to balance, augment and prepare data for training.
- Implement the Neural Network architecture in Keras.

## Inspiration
This project is based on Bojarski et al's paper done with NVIDIA on self-driving using a Convolutional Neural networks. The paper can be found [here](https://arxiv.org/pdf/1604.07316.pdf)

## Related projects
europilot-win.py: https://github.com/Antoine-BL/europilot-win.py
SessionPlayback.py: https://github.com/Antoine-BL/SessionPlayback.py
