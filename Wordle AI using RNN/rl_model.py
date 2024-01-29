import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from collections import deque

wordle_data = pd.read_csv('wordle.csv')
wordle_data.drop('day', axis=1, inplace=True)

class WordleEnvironment:
  def __init__(self, word_list):
    self.word_list = word_list
    self.current_state = np.zeros([6, 5, 3])
    self.target_word = None
    self.max_guesses = 6
    self.guess_count = 0

  def reset(self):
    probabilities = wordle_data['occurrence'] / wordle_data['occurrence'].sum()
    self.current_state = np.zeros([6, 5, 3])
    self.target_word = np.random.choice(self.word_list, p=probabilities)
    self.guess_count = 0
    for i in range(5):
      self.current_state[0][i][2] = ord(self.target_word[i]) - ord('a') + 1
    return self.current_state

  def step(self, action):
    guessed_word = self.word_list[action]
    self.guess_count += 1
    reward = 0 
    
    for i in range(5):
      self.current_state[self.guess_count - 1][i][0] = ord(guessed_word[i]) - ord('a') + 1
      if guessed_word[i] == self.target_word[i]:
        self.current_state[self.guess_count - 1][i][1] = 3
        reward += 1
      elif guessed_word[i] in self.target_word:
        self.current_state[self.guess_count - 1][i][1] = 2
      else:
        self.current_state[self.guess_count - 1][i][1] = 1
        reward -= 1
        
    if guessed_word == self.target_word:
      reward += 10
      reward += 2 * (5 - self.guess_count)
    done = (guessed_word == self.target_word) or (self.guess_count >= self.max_guesses)
    if done and guessed_word != self.target_word:
        reward -= 10
    return self.current_state, reward, done

  def render(self):
    feedback_colors = {0: 'None', 1: 'Grey', 2: 'Yellow', 3: 'Green'}
    
    for i in range(self.guess_count):
      guessed_word_display = []
      for j in range(5):
        letter = chr(int(self.current_state[i][j][0] + ord('a') - 1) if self.current_state[i][j][0] != 0 else '-')
        feedback = feedback_colors[self.current_state[i][j][1]]
        guessed_word_display.append(f"{letter}({feedback})")
      print(f"Target Word: {self.target_word}")
      print(f"Guess {i + 1}: {' '.join(guessed_word_display)}")

word_list = list(wordle_data['word'])
env = WordleEnvironment(word_list)
env.reset()

for a in range(6):
  action = random.randint(0, len(word_list) - 1) 
  state, reward, done = env.step(action)
  env.render()
  print(f"Reward: {reward}\n")
  if done:
    break

def create_q_network(input_dim, output_dim):
  model = Sequential()
  model.add(layers.Dense(128, activation='relu', input_shape=(input_dim,)))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(output_dim))
  model.compile(optimizer='adam', loss='mse')
  return model

class ReplayBuffer:
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
    return state, action, reward, next_state, done

  def __len__(self):
    return len(self.buffer)

class DQNAgent:
  def __init__(self, input_dim, output_dim):
    self.q_network = create_q_network(input_dim, output_dim)
    self.target_network = create_q_network(input_dim, output_dim)
    self.target_network.set_weights(self.q_network.get_weights())
    self.replay_buffer = ReplayBuffer(10000)

  def select_action(self, state, epsilon):
    if random.random() < epsilon:
      return random.randint(0, len(word_list) - 1)
    else:
      state_flat = np.reshape(state, (1, -1))
      q_values = self.q_network.predict(state_flat)
      return np.argmax(q_values[0])

agent = DQNAgent(input_dim=6*5*3, output_dim=len(word_list))

epsilon_initial = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
batch_size = 32
max_episodes = 10000

for episode in range(max_episodes):
  state = env.reset()
  total_reward = 0
  for timestep in range(env.max_guesses):
    epsilon = max(epsilon_final, epsilon_initial * epsilon_decay**episode)
    action = agent.select_action(state, epsilon)
    next_state, reward, done = env.step(action)
    agent.replay_buffer.push(state, action, reward, next_state, done)
    if len(agent.replay_buffer) >= batch_size:
      state_batch, action_batch, reward_batch, next_state_batch, done_batch = agent.replay_buffer.sample(batch_size)
      next_state_batch = np.reshape(next_state_batch, (batch_size, -1))
      q_values_next = agent.target_network.predict(np.array(next_state_batch))
      q_values_next_target = agent.q_network.predict(np.array(next_state_batch))
      target_q_values = reward_batch + (1 - np.array(done_batch)) * 0.99 * np.max(q_values_next_target, axis=1)
      agent.q_network.fit(np.reshape(state_batch, (batch_size, -1)), target_q_values, verbose=0)           
    total_reward += reward
    state = next_state
        
    if done:
      break
            
  print(f"Episode {episode + 1}: Total Reward = {total_reward}")

  if episode % 50 == 0:
    agent.target_network.set_weights(agent.q_network.get_weights())