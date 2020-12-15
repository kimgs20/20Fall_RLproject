import numpy as np
import random
from sarsa_environment import SarsaEnv
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

# hyperparam
num_episode = 1000
# step_size = 0.01
# discount_factor = 0.9
# epsilon = 0.1

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01 # step_size = learning rate
        self.discount_factor = 0.9
        self.epsilon = 0.1
        # 0을 초기값으로 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q       

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)


episode_reward = []
if __name__ == "__main__":
    env = SarsaEnv()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(num_episode):
        state = env.reset()
        action = agent.get_action(state)

        eps_reward = 0
        while True:
            
            # 게임 환경과 상태를 초기화
            env.render()
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
            next_state, reward, done = env.step(action)

            # next_action = agent.get_action(str(next_state))
            next_action = agent.get_action(next_state)

            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(state, action, reward, next_state, next_action)
            # agent.learn(str(state), action, reward, str(next_state), next_action)

            ##################
            # What is Next Action???

            state = next_state
            action = next_action
            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)
            
            eps_reward += reward
            print(episode+1, eps_reward)
            # plt.plot()
            

            if done:
                episode_reward.append(eps_reward)
                # plot_episodes():
                break


eps_list = []
for i in range(num_episode):
    eps_list.append(i+1)

my_dpi = 100
plt.figure(figsize=(1600/my_dpi, 800/my_dpi), dpi=my_dpi)

# plt.figure()
plt.plot(eps_list, episode_reward)
plt.title("Cliff Walking: SARSA")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('./Sarsa_default.png')
plt.ioff()
plt.show()