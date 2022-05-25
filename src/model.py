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
        # 8 x 8 x 4 with 32 Filters ,Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
        self.conv1=nn.Conv2d(4,32,kernel_size=8,stride=4,padding = 2)
        self.mp1=nn.MaxPool2d(2,stride=2,padding=1)  #2x2x32 stride=2
        # 4 x 4 x 32 with 64 Filters ,Stride 2 -> Output 6 x 6 x 64
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2,padding=2)
        # 3 x 3 x 64 with 64 Filters,Stride 1 -> Output 6 x 6 x 64
        self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1,padding = 1)

        self.fc1=nn.Linear(2304,4)
     
    def forward(self,x):
            
        x=F.relu(self.conv1(x))
        x=self.mp1((x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))     
        conv3_to_reshaped=torch.reshape(x,[-1,2304])
        output=self.fc1(conv3_to_reshaped)
        return output  #action's value


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

    def train_step(self, state, action, reward, next_state, done):
        #state = torch.tensor(state, dtype=torch.float)
        #next_state = torch.tensor(next_state, dtype=torch.float)
        #action = torch.tensor(action, dtype=torch.long)
        #reward = torch.tensor(reward, dtype=torch.float)
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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.learn_step_counter += 1

        self.optimizer.step()



