import random


#action is two numbers, an x coordinate followed by 
class Random:
   def  __init__(self, board_width):
        #parameters go here

        self.last_state = None
        self.board_width  = board_width
        self.bounding_box_width = 3
        

   def interact(self, state, reward):

        self.last_state = state
        return [random.randrange(self.board_width - self.bounding_box_width), random.randrange(4)]