from agents import *
from task import TetrisTask
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def fittedQ(task, explore_agent = random, trials = 1000, 
    regressor = ExtraTreesRegressor, regressor_params = {}, N = 30,
    prev_sample = None):

    Q = None
    n = 0

    def q_prediction(sa):
        return Q.predict(sa) if Q else 0

    state_histories, action_histories, reward_histories = task.run_trials()
    

    #todo add previous sample support
    # if prev_sample:
    #     state_histories = np.concatenate((prev_sample[0], state_histories ))
    #     action_histories = np.concatenate((prev_sample[1], action_histories))
    #     reward_histories = np.concatenate((prev_sample[2], reward_histories))

    # patients = len(state_histories)
    print state_histories[0]
    print action_histories[0]
    print reward_histories[0]

    while (n < N):
    	total_turns = sum([len(game) for game in state_histories])
    	training_data = np.zeros()


    #generate function approximator
    while (n < N) :
        
        #build training set
        training_data = np.zeros((patients*episode_length, state_size+1))
        training_targets = np.zeros(patients*episode_length)
        for patient in range(patients):
            for episode in range(episode_length):
                state_action = np.append(state_histories[patient][episode], action_histories[patient][episode]) 

                next_sa = [np.append(state_histories[patient][episode+1], action) for action in xrange(num_actions) ]
                q = reward_histories[patient][episode] + discount * max([q_prediction(sa) for sa in next_sa])
                
                #convert sa here
                training_data[patient*episode + episode] = state_action
                training_targets[patient*episode + episode] = q

        #train regressor
        Q = regressor(**regressor_params)
        Q.fit(training_data, training_targets)

        n += 1


    #get policy
    def learned_policy(state, rng):
        next_sa = [np.append(state, action) for action in xrange(num_actions) ]
        action = next_sa[np.argmax([q_prediction(sa) for sa in next_sa])][-1]
        return action

    return learned_policy, (state_histories, action_histories, reward_histories), Q
