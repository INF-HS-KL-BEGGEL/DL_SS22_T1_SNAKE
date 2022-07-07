import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
from pathlib import Path


from game import SnakeGameAI

from metrics import MetricLogger
from agent import SnakeAgent

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
maxscore = 0

current_checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/snake.chkpt')
agent = SnakeAgent(state_dim=(4, 84, 84), action_dim=4, save_dir=save_dir, checkpoint=current_checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000
game = SnakeGameAI()
### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):
    state = game.initState()
    # Play the game!
    while True:

        action, calc, rand = agent.get_action(state)

        # 5. Agent performs action
        #### funktion play_step in game anpassen.

        next_state, reward, done, score = game.play_step(action)
        maxscore = max(maxscore, score)
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
        print("Random Action: %s, Calc Action: %s, Score: %s" % (rand, calc, maxscore) )
        agent.resetCounter()
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )