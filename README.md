# Full-Duplex-User-Pairing_1
This project is about user pairing in a full-duplex communication system based on the deep reinforcement learning (DQN). We use 32 different groups of users to train the agent, each group include 6 uplink users and 8 downlink users. First, we transform the problem of user pairing to the markov decision process. Then, to accelerate the training speed, we added some expert experience to the replay buffer to make the agent learn quikly.
