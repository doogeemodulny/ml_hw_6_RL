import torch
import numpy as np
import random
import time
import torch.nn as nn
import torch.optim as optim
import neptune

from game.wrapped_flappy_bird import GameState
from preprocessing import resize_and_bgr2gray, image_to_tensor

from config import (DEVICE, GAMMA, START_EPSILON, FINAL_EPSILON, 
                   NUMBER_OF_ITERATIONS, REPLAY_MEMORY_SIZE, 
                   MINIBATCH_SIZE, NUMBER_OF_ACTIONS, LEARNING_RATE)

class Trainer:
    def __init__(self,
                model,
                gamma=GAMMA,
                start_epsilon=START_EPSILON,
                final_epsilon=FINAL_EPSILON,
                number_of_iterations=NUMBER_OF_ITERATIONS,
                replay_memory_size=REPLAY_MEMORY_SIZE,
                minibatch_size=MINIBATCH_SIZE):
        self.model = model
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.number_of_iterations = number_of_iterations
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size

        self.number_of_actions = NUMBER_OF_ACTIONS
        self.start_time = time.time()
        
        self.run = neptune.init_run(
            project="ve1llon/ml-homework6",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZGNiNDkyNS03NjlhLTQ2YzYtODZmYy1kYTQyMWRkMWIzODUifQ=="
        )

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)

    def _initialize_weights(self):
        for m in self.model.modules():  
            self.init_weights(m)        

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        game_state = GameState(is_model=True)

        replay_memory = []
        action = torch.zeros([self.number_of_actions], dtype=torch.float32)
        action[0] = 1
        image_data, reward, terminal, score = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        epsilon = self.start_epsilon
        iteration = 0
        self.start_time = time.time()
        score_record = 0

        epsilon_decrements = np.linspace(self.start_epsilon, self.final_epsilon, self.number_of_iterations)

        while iteration < self.number_of_iterations:
            output = self.model(state)[0]

            action = torch.zeros([self.number_of_actions], dtype=torch.float32).to(DEVICE)

            random_action = random.random() <= epsilon
            if random_action:
                print("Performed random action!")
            action_index = [torch.randint(self.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action  # exploration
                            else torch.argmax(output)][0]  # exploitation
            action_index.to(DEVICE)

            action[action_index] = 1

            image_data_1, reward, terminal, score = game_state.frame_step(action)
            image_data_1 = resize_and_bgr2gray(image_data_1)
            image_data_1 = image_to_tensor(image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            replay_memory.append((state, action, reward, state_1, terminal))

            if len(replay_memory) > self.replay_memory_size:
                replay_memory.pop(0)

            epsilon = epsilon_decrements[iteration]

            minibatch = random.sample(replay_memory, min(len(replay_memory), self.minibatch_size))

            state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(DEVICE)
            action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(DEVICE)
            reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(DEVICE)
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).to(DEVICE)

            output_1_batch = self.model(state_1_batch)

            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))

            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)
            optimizer.zero_grad()
            y_batch = y_batch.detach()
            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()

            state = state_1
            iteration += 1
            score_record = max(score, score_record)

            if iteration % 100000 == 0:
                torch.save(self.model, "current_model_" + str(iteration) + ".pth")

            if iteration % 100 == 0:
                self.run["metrics/loss"].log(loss.item())
                self.run["metrics/epsilon"].log(epsilon)
                self.run["metrics/reward"].log(reward.item())
                self.run["metrics/action"].log(action_index.cpu().item())
                self.run["metrics/Q_max"].log(output.max().item())
                self.run["metrics/score_record"].log(score_record)
        self.run.stop()