
# A Reinforcement-learning Agent learns how to play the classic game Snake

As part of a master project, a functional snake agent was developed with the help of [Double Q-learning](https://towardsdatascience.com/double-deep-q-networks-905dd8325412), [PyTorch](https://pytorch.org) and a [CNN](https://de.wikipedia.org/wiki/Convolutional_Neural_Network). 

The Blogpost (german) of this project can be found here.

## Setup

 1. Install [Python](https://www.python.org), [Pip](https://pypi.org/project/pip/), [cuda](https://developer.nvidia.com/cuda-downloads) and [VirtualEnvironment](https://docs.python.org/3/tutorial/venv.html)
 2. create a virtualenviroment with `python3 -m venv /path/to/new/virtual/environment`
 3. activate the enviroment (unix) `source <venv>/bin/activate` (windows) `C:\> <venv>\Scripts\activate.bat` you should see a (venv) in front of your command prompt
 4. install dependencys with`pip install -r requirements.txt`

## Running
To start the  **learning**  process for the Snake,

    python main.py

This starts the  _double Q-learning_  and logs the training data to the  `checkpoints` folder.

You can also train an existing model by specifying a checkpoint in the `main.py` class.  e.g `current_checkpoint = Path(''checkpoints/timestamp/name_of_the_checkpoint.chkpt')`

GPU will automatically be used if available.

To  **evaluate**  a trained Snake-model use,

    python replay.py

This starts a window in which the Snake-agent "plays" a trained model. No training will take place. To evaluate a specific model, simply change the save_dir in `replay.py`. 
E.g `checkpoint = Path('checkpoints/timestamp/name_of_the_checkpoint.chkpt')`

## Pre-Trained
A pretrained model can be found in the `src/pretrained_checkpoint` folder.  The model was trained for approx. 24 hours. A Google Cloud VM instance was used as the training hardware:
 - n1-highmem-2
 - Tesla K 80 GPU

The pre-trained model achieved the following scores:
 - max. Score: 28
 - avg. Score: 12

## Project Structure
- agent: communicates with game and neural, calculates training values, stores experiences and controlls learning
- game: the snake game that the agent will use
- main: used to start the training process
- neural: This class defines the neural network
- replay: used to let the agent play games with a pre trained Net (No learning will be done)
- metrics: Define a `MetricLogger` that helps track training/evaluation performance.

## Ressourcen/Quellen
-   [Deep Learning Flappy-Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)
-   [Snake-Ai-Pytorch](https://github.com/python-engineer/snake-ai-pytorch)
-   [Screenshot Methode](https://github.com/benjamin-dupuis/DQN-snake)
 -   [DDQN](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)
  -   [Exploitation vs Exploration](https://www.baeldung.com/cs/epsilon-greedy-q-learning)
  -    [Mad Mario]( https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
  - [Double Q-Learning](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)
  - [Q-Learning](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
  - [epsilon-greedy-Prinzip](https://towardsdatascience.com/exploration-in-reinforcement-learning-e59ec7eeaa75)
  - [Loss](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)
  - [Q-values](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)
  -   [Reward](https://towardsdatascience.com/how-to-design-reinforcement-learning-reward-function-for-a-lunar-lander-562a24c393f6)
  - [deque](https://docs.python.org/3/library/collections.html#collections.deque)
  - [Reinforcement-Learning](https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/)
  - [PyGame](https://www.pygame.org/)
  - [ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
  - [Dense-Layer](https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/)
  - [Convolutional-Layer](https://www.sciencedirect.com/topics/engineering/convolutional-layer)
  
- [Start Videoreihe zum Thema](https://www.youtube.com/watch?v=PJl4iabBEz0)

- [Paper hs-albsig](https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/)

- [Paper towardsdatascience](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)

- [Paper Openreview.net siehe PDF Ordner](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwics97z1q73AhUKQvEDHRGvDGgQFnoECA8QAQ&url=https%3A%2F%2Fopenreview.net%2Fpdf%3Fid%3Diu2XOJ45cxo&usg=AOvVaw0zAkY431TzL4zegennRyqX)

- [Paper researchgate.net](https://www.researchgate.net/publication/351884746_A_Deep_Q-Learning_based_approach_applied_to_the_Snake_game)

- [Paper geeksforgeeks](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/)
