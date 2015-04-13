import random


#action is two numbers, an x coordinate followed by a rotation 

class Random:
    #TODO implement valid_actions that returns the set of valid actions
   def  __init__(self, board_width):
        #parameters go here

        self.last_state = None
        self.board_width  = board_width
        self.bounding_box_width = 3
        

   def interact(self, state, reward):

        self.last_state = state
        return [random.randrange(self.board_width - self.bounding_box_width), random.randrange(4)]


class PolicyAgent:
    def  __init__(self, policy):

        self.policy = policy

    def interact(state, reward):
        #determine using policy
        return 0
