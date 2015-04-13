import random


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


class PolicyAgent:
    def  __init__(self, policy):

        self.policy = policy

    def interact(state, reward):
        #determine using policy
        return 0
