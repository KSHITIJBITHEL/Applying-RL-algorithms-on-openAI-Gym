# This is the implementation of tabular q learning algorithm for the Mountain Car gym env
import gym
import numpy as np
env = gym.make("MountainCar-v0")
env.reset()
# Q-Learning hyperparameters
LEARNING_RATE = 0.95 #the weight of the newly learned q value over prev one
DISCOUNT = 0.95 #the weight of the future rewards w.r.t immediate reward
EPISODES = 2000
SHOW_EVERY = 500   
epsilon = 1 
epsilon_min = 0.05
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value =(epsilon-epsilon_min)/(END_EPSILON_DECAYING - START_EPSILON_DECAYING) 
DISCRETE_OS_SIZE = [50,50] #discretizing the obs space [pos vel] to reduce computation
discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

#Initialise the q table with random values
q_table = np.random.uniform(low = -2,high =0, size= (DISCRETE_OS_SIZE+ [env.action_space.n]))

#Generate discrete states for reducing computation and size of q_table
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode  in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    if episode % SHOW_EVERY == 0:
        render= True
        print(episode)
    else:
        render = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) # Get action from Q table
        else:
            action = np.random.randint(0, env.action_space.n)  # Get random action
        new_state,reward,done,_ = env.step(action)
        new_discrete_state =get_discrete_state(new_state)
        if(render):
            env.render()
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
        # Simulation ended (for any reason) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = reward
                print("reached goal at",episode)
        discrete_state= new_discrete_state
         # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

env.close()