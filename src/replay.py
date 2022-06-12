import random, datetime
from pathlib import Path

from metrics import MetricLogger
from agent import SnakeAgent
from game import SnakeGameAI

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-06-09T21-06-58/snake_cnn10.chkpt')
snakeagent = SnakeAgent(state_dim=(4, 84, 84), action_dim=3, save_dir=save_dir, checkpoint=checkpoint)
snakeagent.exploration_rate = snakeagent.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 1000
game = SnakeGameAI()

for e in range(episodes):
    game.reset()
    observation = game.screenshot()
    state = game.get_last_frames(observation)

    while True:

        action = snakeagent.act(state)

        next_state = game.get_last_frames(game.screenshot())
        reward, done, score = game.play_step(action)

        snakeagent.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done:
            game.reset()
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=snakeagent.exploration_rate,
            step=snakeagent.curr_step
        )
