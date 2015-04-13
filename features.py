# return the location of the top of the stack (the first blank row)
def get_top(board):
  top = len(board) - 1
  while not all([x == '' for x in board[top]]) and top > 2:
    top -= 1
  return top

# return the top four rows as heights 0 - 4
def get_top_four(board):
  top = get_top(board)
  bottom = len(board) - 1
  width = len(board[0])
  heights = [0] * width
  rows = min(4, bottom - top)
  for i in reversed(range(rows)):
    for j in range(width):
      if board[top+i+1][j] != '':
        heights[j] = rows - i
  return heights    
  
# return the number of holes in the board
def get_num_holes(board):
  top = get_top(board)
  height = len(board)
  width = len(board[0])
  count = 0
  for i in range(top+1, height):
    for j in range(width):
      if i-1 in range(height) and board[i][j] == '' and board[i-1][j] != '':
        count += 1
  return count

# encode the current piece as a binary vector
def get_tet(tet):
  pieces = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']
  if tet == None:
    return
  else:
    return [int(tet == piece) for piece in pieces]

def get_features(board, tet):
  if type(board) is bool:
    return
  else:
    return get_top_four(board) + [get_num_holes(board)] + get_tet(tet)