import random
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import numpy as np

#action is two numbers, an x coordinate followed by a rotation 
pieces = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']


def state_to_tet(state):
    return pieces[list(state[-7:]).index(1)]

def get_valid_actions(tet):
    bounding_box_width = 3
    board_width = 8
    leftmost = board_width - bounding_box_width + 1
    # +1 to compensate for exclusive bound of range function

    #square
    if tet == 'O': 
        return [[c, 0] for c in range(-1, leftmost)]

    #line
    if tet == 'I': 
        return [[c, 0] for c in range(0, leftmost-1)] 
        + [[c, 1] for c in range(-2, leftmost)] 

    #left column free on rotation 1, right column free on rotation 3
    else:
        return [[c,r] for c in range(0, leftmost) for r in [0,2]]
        + [[c,1] for c in range(-1, leftmost)]
        + [[c,3] for c in range(0, leftmost+1)]

valid_actions = {}
for p in pieces:
    valid_actions[p] = get_valid_actions(p)



class Random:
    #TODO implement valid_actions that returns the set of valid actions
    def  __init__(self, board_width):
        #parameters go here

        self.last_state = None
        self.board_width  = board_width

    def interact(self, state, reward, field, tet):

        self.last_state = state
        valid = valid_actions[tet]
        action = random.choice(valid)
        return action

class FittedQAgent:
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        self.regressor_params = regressor_params
        self.reg = self.regressor(**self.regressor_params)
        r = Random(board_width)
        self.random = r
        self.current_policy = r.interact
        self.pieces = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']

        self.last_state = None
        self.last_action = None
        self.tuples = []
        self.n_tuples = 0

    def interact(self, state, reward, field, tet):
        if self.last_state != None and state != None:
            self.tuples.append((self.last_state, self.last_action, reward, state))
            self.n_tuples += 1
            if self.n_tuples == self.n_samples:
                self.regress()
                self.n_tuples = 0
        self.last_state = state
        self.last_action = self.current_policy(state, reward, field, tet)
        return self.last_action


    def regress(self):
        print 'regressing on %s tuples' % len(self.tuples)


        data = np.array([s+a for (s, a, r, new_s) in self.tuples])
        rewards = np.array([r for (s, a, r, new_s) in self.tuples])
        targets = np.array([r for (s, a, r, new_s) in self.tuples])

        new_data = ([new_s for (s, a, r, new_s) in self.tuples])

        reg = self.reg
        for i in range(self.N):
            reg.fit(data, targets)

            for j in range(len(data)):
                state = new_data[j]
                next_sa = np.array([state + a for a in  valid_actions[state_to_tet(state)]])
                targets[j] = rewards[j] + self.gamma * np.amax(reg.predict(next_sa))

        def learned_policy(state, reward, field, tet):
            if not state is None:
                next_sa = [state+a for a in valid_actions[state_to_tet(state)] ]
                action = next_sa[np.argmax([reg.predict(sa) for sa in next_sa])][-2:]
                return action

        self.current_policy = learned_policy
        #random.shuffle(self.tuples)
        #self.tuples = self.tuples[:(self.n_samples*4/5)]
        print 'done'

class PolicyAgent:
    def  __init__(self, policy):

        self.policy = policy

    def interact(self, state, reward, field, tet):
        #determine using policy
        return self.policy(state, reward, field, tet)
