from task import TetrisTask
from agents import *

board_width = 8
agent = Random(board_width)
task = TetrisTask(agent, width = 8, height = 22)

task.run_trials(2)