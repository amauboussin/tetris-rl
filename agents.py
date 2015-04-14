import random
import numpy as np

#action is two numbers, an x coordinate followed by a rotation 

class Random:
    #TODO implement valid_actions that returns the set of valid actions
    def  __init__(self, board_width):
        #parameters go here

        self.last_state = None
        self.board_width  = board_width
        self.bounding_box_width = 3
    
    def valid_actions(self, field, tet):

        leftmost = self.board_width - self.bounding_box_width + 1
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


    def interact(self, state, reward, field, tet):

        self.last_state = state
        valid = self.valid_actions(field, tet)
        action = random.choice(valid)
        return action

class FittedQAgent:
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(N, gamma, board_width, n_samples, regressor):
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        r = Random(board_width)
        self.random = r
        self.current_policy = r.interact
        self.pieces = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']

        self.last_state = None
        self.last_action = None
        self.tuples = []

    def interact(self, state, reward, field, tet):
        if self.last_state != None:
            self.tuples.append((self.last_state, self.last_action, reward, state))
            if len(self.tuples) == n_samples
                self.regress()
        self.last_state = state
        self.last_action = self.current_policy(state, reward, field, tet)
        return self.last_action

    def regress():
        reg = self.regressor()
        reg_input = np.array([s+a for (s, a, r, new_s) in self.tuples])
        reg_output = np.array([r for (s, a, r, new_s) in self.tuples])
        N = self.N
        while(N > 0):
            reg.fit(reg_input, reg_output)
            tet =

class PolicyAgent:
    def  __init__(self, policy):

        self.policy = policy

    def interact(state, reward):
        #determine using policy
        return 0
