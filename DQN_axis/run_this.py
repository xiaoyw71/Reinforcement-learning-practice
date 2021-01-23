from study.DQN.maze_env import Maze
#from study.DQN.RL_brain import DeepQNetwork
from study.DQN.DQN_modified import DeepQNetwork

def run_maze():
    step = 0 
    for episode in range(300):
        # initial observation
        observation = env.reset()
        #observation = env.get_image()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # 取当前窗口为图片numpy
            #observation_ = env.get_image()

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    
    RL.plot_cost()