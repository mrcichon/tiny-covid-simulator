import torch
import torch.nn as nn
import pandas as pd
from collections import namedtuple, deque
import random

"""
A dataset of field-states paired with 'correct' next move(coordinates) for a given agent will be needed.
A correct move is a move that will result in the agent avoiding infection.
There doesn't seem to be any need for a complex architecture, for now.
Should simple shallow, fully-connected network fail, a transformer network will be used.
The model should at first indicate in which 'direction' the agent should move.
Reinforced learning also seems like a viable idea.
Maybe XGBoost? It *is* kinda tabular data
"""

df = pd.read_csv('sight.csv', on_bad_lines='skip')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sort_agents():
    df_by_id = []
    for ID in df['ID'].unique():
        df_by_id.append(df[df['ID'] == ID])
    return df_by_id

def label_data(df_by_id):
    for agent in df_by_id:
        if agent['Group'] == 'Suspectible':
            agent['Label'] = 1
        elif agent['Group'] == 'Infected':
            agent['Label'] = 0
        elif agent['Group'] == 'Immune':
            df_by_id.drop(agent)
    return df_by_id



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# skeleten code for reinforced learning
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# skeleten code for reinforced learning part II
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# skeleton code for basic feedfoward net
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.architecture = nn.Sequential(
            nn.Linear(100, 10),
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    def foward(self, x):
        x = self.flatten(x)
        logits = self.architecture(x)
        return x


model = SimpleNet().to(device)
print(model)
