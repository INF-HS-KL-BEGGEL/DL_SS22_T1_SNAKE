import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

from game import SnakeGameAI

from metrics import MetricLogger
from agent import SnakeAgent

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/snake.chkpt')
agent = SnakeAgent(state_dim=(4, 84, 84), action_dim=4, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000
game = SnakeGameAI()
### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):
    game.reset()
    observation = game.screenshot()
    state = game.get_last_frames(observation)
    # Play the game!
    while True:

        action = agent.act(state)

        # 5. Agent performs action
        #### funktion play_step in game anpassen.
        next_state = game.get_last_frames(game.screenshot())
        reward, done = game.play_step(action)

        # 6. Remember
        agent.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = agent.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done:
            game.reset()
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )