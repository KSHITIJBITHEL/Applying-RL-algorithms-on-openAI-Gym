import gym 
import numpy as np
import matplotlib.pyplot as plt
#create an instance of the env 
env_name = "CartPole-v1"
env = gym.make(env_name)
env.reset()
''' Observation Space:
        0 : Cart Position
        1 : Cart Velocity
        2 : Pole angle
        3 : Pole Velocity at tip
    Action Space :
        0 : push cart to left 
        1 : push cart to right 
        Note: The amount the velocity is reduced or increased is not fixed 
        as it depends on the angle the pole is pointing. 
        This is because the center of gravity of the pole increases 
        the amount of energy needed to move the cart underneath it
    Reward : 
        Reward is 1 for every step taken, including the termination step
    Episode Termination :
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 500 .
'''

# Q-Learning settings
LEARNING_RATE = 0.9 #the weight of the newly learned q value over prev one
DISCOUNT = 0.5 #the weight of the future rewards w.r.t immediate reward
EPISODES = 50000
SHOW_EVERY = 500   
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES- 1000
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING) 
epsilon_min = 0.08
obs_high = [4.8 ,5 ,0.42 ,3]
obs_low = [-4.8 ,-5 ,-0.42 ,-3]
DISCRETE_OS_SIZE = [100,100,100,100] #discretising the obs space to reduce computation
diff = [i-j for i,j in zip(obs_high,obs_low)] #[9.6, 10, 0.84, 4]
discrete_os_win_size = [i / j for i, j in zip( diff, DISCRETE_OS_SIZE)] #[0.192, 0.2, 0.0168, 0.08]
q_table = np.random.uniform(low = -2,high =0, size= (DISCRETE_OS_SIZE+ [env.action_space.n]))#Initialise the q table(50, 50, 50, 50, 2) with random values

class Agent():
        def __init__(self,env):
                self.action_size = env.action_space.n 
                print('action size:',self.action_size)
        def get_action(self, state,epsilon):
                if np.random.random() > epsilon:
                        action = np.argmax(q_table[descrete_state]) # Get action from Q table
                else:
                        action = np.random.randint(0, env.action_space.n)# Get random action
                return action 
        #Generate discrete states for reducing computation and size of q_table
        def get_discrete_state(self,state):
                diff = [i-j for i,j in zip(state,obs_low)] 
                discrete_state = [i // j for i, j in zip( diff, discrete_os_win_size)] 
                for i in range(4):
                        if(int(discrete_state[i]) >= 99 ):
                                discrete_state[i] =99
                return tuple(int(i) for i in discrete_state)  # we use this tuple to look up the 3 Q values for the available actions in the q-table
avg_return = 0
agent = Agent(env)
reward_plot =[]
for episode in range(EPISODES):
        done = False
        cum_reward = 0
        state = env.reset()   
        descrete_state = agent.get_discrete_state(state)
        # print(descrete_state)
        if episode % SHOW_EVERY == 0 and episode> EPISODES/10.0:
                render= False
                print(episode,epsilon)
        else:
                render = False
        while not done:
                action = agent.get_action(descrete_state,epsilon)
                # print('action',action)
                new_state,reward,done,_ = env.step(action)
                # print('new_state',new_state,'reward', reward, done)
                new_discrete_state =agent.get_discrete_state(new_state)
                # print(new_discrete_state)
                if(render):
                        env.render()
                # If simulation did not end yet after last step - update Q table
                if not done:
                        max_future_q = np.max(q_table[new_discrete_state])# Maximum possible Q value in next step (for new state)
                        current_q = q_table[descrete_state + (action,)]# Current Q value (for current state and performed action)
                        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)# new Q value for current state and action
                        q_table[descrete_state + (action,)] = new_q# Update Q table with new Q value
                descrete_state= new_discrete_state
                cum_reward += reward
        if(epsilon>epsilon_min):
                epsilon -= epsilon_decay_value
        if (episode%40 ==0):
                reward_plot.append(avg_return) 
                avg_return = 0
        else:
                avg_return += cum_reward/40
x_axis = np.arange(np.size(reward_plot))
plt.plot(x_axis,reward_plot)
plt.show() 
print('done')             
env.close()