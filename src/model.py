from turtle import shape
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as tv
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Conv2d(84, 84, kernel_size=4, stride=4)
        self.linear2 = nn.Conv2d(84, 16, kernel_size=4, stride=2)

        self.dense1 = torch.nn.Linear(3872,256)
        self.dense2 = torch.nn.Linear(256, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = torch.reshape(x, shape=[-1, 32*11*11])
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, replace):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = model
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("test")

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        self.replace_target_network()
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )


        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.learn_step_counter += 1

        self.optimizer.step()



