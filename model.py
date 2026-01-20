import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state_old, final_move, reward, state_new, done):
        # Convert to tensors
        state_old = torch.tensor(state_old, dtype=torch.float)
        state_new = torch.tensor(state_new, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        final_move = torch.tensor(final_move, dtype=torch.long)
        done = torch.tensor(done, dtype=torch.bool)

        # If single sample, add batch dimension
        if len(state_old.shape) == 1:
            state_old = state_old.unsqueeze(0)
            state_new = state_new.unsqueeze(0)
            reward = reward.unsqueeze(0)
            final_move = final_move.unsqueeze(0)
            done = done.unsqueeze(0)

        # Current Q-values for state_old
        pred = self.model(state_old)  # shape: (batch, 4)

        # Build target = pred, but replace chosen action with updated Q
        target = pred.clone().detach()

        with torch.no_grad():
            next_q = self.model(state_new)  # (batch, 4)
            max_next_q = torch.max(next_q, dim=1).values  # (batch,)

        action_idx = torch.argmax(final_move, dim=1)  # (batch,)

        # Q_new = r if done else r + gamma * max(Q_next)
        q_new = reward + (~done).float() * (self.gamma * max_next_q)

        target[torch.arange(target.size(0)), action_idx] = q_new

        # Optimize
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
