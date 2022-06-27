import random, datetime
from pathlib import Path

from metrics import MetricLogger
from agent import SnakeAgent
from game import SnakeGameAI

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-06-09T21-06-58/snake_cnn5.chkpt')
snakeagent = SnakeAgent(state_dim=(4, 84, 84), action_dim=4, save_dir=save_dir, checkpoint=checkpoint)
snakeagent.exploration_rate = snakeagent.exploration_rate_min

logger = MetricLogger(save_dir)
maxscore = 0

episodes = 1000
game = SnakeGameAI()

for e in range(episodes):
    state = game.initState()
    while True:

        action, calc, rand = snakeagent.get_action(state)
        next_state, reward, done, score = game.play_step(action)
        snakeagent.cache(state, next_state, action, reward, done)
        maxscore = max(maxscore, score)

        logger.log_step(reward, None, None)

        state = next_state

        if done:
            game.reset()
            break

    logger.log_episode()

    if e % 20 == 0:
        print("Maxscore: %s" % maxscore)
        logger.record(
            episode=e,
            epsilon=snakeagent.exploration_rate,
            step=snakeagent.curr_step
        )
