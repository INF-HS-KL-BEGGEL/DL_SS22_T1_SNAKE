# Ein Reinforcement-Learning Agent lernt den Spieleklassiker Snake 

## Setup


## Pre-Trained
Einen vor trainierten Checkpoint kann im Ordner preTrained gefunden werden

## Projektaufbau Structure
- agent: communicates with game and neural, calculates training values, stores experiences and controlls learning
- game: the snake game that the agent will use
- main: used to start the training process
- neural: This class defines the neural net
- replay: used to let the agent play games with a pre trained Net (No learning will be done)
- metrics: from  [Mad Mario]( https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) Define a `MetricLogger` that helps track training/evaluation performance.

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
  
Weitere gefundene Dateien/Videos:

-[Start Videoreihe zum Thema](https://www.youtube.com/watch?v=PJl4iabBEz0)

-[Paper hs-albsig](https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/)

-[Paper towardsdatascience](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)

-[Paper Openreview.net siehe PDF Ordner](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwics97z1q73AhUKQvEDHRGvDGgQFnoECA8QAQ&url=https%3A%2F%2Fopenreview.net%2Fpdf%3Fid%3Diu2XOJ45cxo&usg=AOvVaw0zAkY431TzL4zegennRyqX)

-[Paper researchgate.net](https://www.researchgate.net/publication/351884746_A_Deep_Q-Learning_based_approach_applied_to_the_Snake_game)

-[Paper geeksforgeeks](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/)
