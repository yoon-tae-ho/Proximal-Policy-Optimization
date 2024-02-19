# 필요한 라이브러리와 모듈 임포트
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from PPO import PPO
from utils import env_reset, env_next_step

random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num

def test():
    env_name = "young-car"
    env_path = f"./{env_name}"

    # 하이퍼파라미터 설정
    has_continuous_action_space = True
    max_ep_len = 5000
    action_std = 0.1
    total_test_episodes = 3
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
  
    #####################################################
  
    # 그래프를 위한 데이터 저장소 초기화
    state_values = [[], [], []]  # 각각 Longitudinal Velocity, Longitudinal Acceleration, Vertical Acceleration에 해당

    # 그래프 설정
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    lines = [ax.plot([], [])[0] for ax in axs]
    titles = ['Longitudinal Velocity', 'Longitudinal Acceleration', 'Vertical Acceleration']
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_xlim(0, max_ep_len)
        ax.set_ylim(-5, 5)  # 이 값은 상태에 따라 조정될 필요가 있습니다.

    plt.ion()
    plt.show()
  
    #####################################################

    # Unity Environment
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_path, side_channels=[engine_configuration_channel], seed=random_seed)
    env.reset()

    # Unity Brain
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60)

    # state space dimension
    state_dim = spec.observation_specs[0].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = spec.action_spec.continuous_size
    else:
        action_dim = spec.action_spec.discrete_size
  
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
  
    # preTrained weights directory
    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env_reset(env, behavior_name)

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            action = np.expand_dims(action, axis=0)
            state, reward, done = env_next_step(env, behavior_name, action)
            
            reward = np.squeeze(reward, axis=0)
            ep_reward += reward

            # 상태 값 업데이트 (여기서는 예시로 3개의 특정 상태를 사용)
            state_values[0].append(state[0][2])  # Longitudinal Velocity
            state_values[1].append(state[0][4])  # Longitudinal Acceleration
            state_values[2].append(state[0][5])  # Vertical Acceleration
            
            if t % 10 == 0:  # 주기적으로 그래프 업데이트
                for i, line in enumerate(lines):
                    line.set_data(range(len(state_values[i])), state_values[i])
                plt.draw()
                plt.pause(0.001)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    test()
    plt.show(block=True)