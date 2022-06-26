from os import stat
import torch
import random, numpy as np
from pathlib import Path

from neural import SnakeCNN
from neural import NetMode
from collections import deque

MAX_MEM_SIZE=40000

class SnakeAgent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MAX_MEM_SIZE)
        self.batch_size = 32
        self.random = 0
        self.calc = 0

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.steps_before_learning = MAX_MEM_SIZE  # Anzahl an Schritte bevor Lernvorgang beginnt
        self.learn_every = 3   #Trainingsnetz wird alle 3 Schritte aktualisiert
        self.sync_every = 1e4   # Anzahl Schritte bevor target Netzwerk aktualisiert wird

        self.save_every = 1e5   #Anzahl Schritte nach denen gespeichert wird
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available() #kann Grafikkarte verwendet werden?

        self.net = SnakeCNN(self.state_dim, self.action_dim).float()    #enthällt target und Zielnetz. kann über enum ausgewählt werden
        if self.use_cuda:       #Nutze Grafikkarte für Netz wenn möglich. Ansonsten cpu
            self.net = self.net.to(device='cuda')
        if checkpoint:      #lade Checkpoint wenn vorhanden
            self.load(checkpoint)

        print(self.use_cuda)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_function = torch.nn.SmoothL1Loss()


    def act(self, state):
        # EXPLORE (Ausführen zufälliger Actionen um Gedächtniss zu füllen)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
            self.random += 1

        # EXPLOIT   (verwendet Voraussage der KI)
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model=NetMode.TRAINING)
            action_idx = torch.argmax(action_values).item()
            self.calc +=1

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx, self.calc, self.random

    def resetCounter(self):
        self.calc = 0
        self.random = 0

    def cache(self, state, next_state, action, reward, done):

        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    @torch.no_grad()
    def calc_next_Q(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model=NetMode.TRAINING)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model=NetMode.TARGET)[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_training(self, current_Q, next_Q) :
        loss = self.loss_function(current_Q, next_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.targetNet.load_state_dict(self.net.trainingNet.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.steps_before_learning:
            return None, None
        elif self.curr_step == self.steps_before_learning:
            print("--------start training--------")

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get Estimate
        estimate =self.net(state, model=NetMode.TRAINING)[np.arange(0, self.batch_size), action]

        # Get next Q for loss
        nextQ = self.calc_next_Q(reward, next_state, done)

        loss = self.update_Q_training(estimate, nextQ)

        return (estimate.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"snake_cnn{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"SnakeCNN saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        checkpoint_to_load = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        self.exploration_rate = checkpoint_to_load.get('exploration_rate')
        state_dict = checkpoint_to_load.get('model')
        self.net.load_state_dict(state_dict)
        print(f"Loading model at {load_path} with exploration rate {self.exploration_rate}. Cuda enabled? {self.use_cuda}")


