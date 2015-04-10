from random import shuffle


class TetrisRandomGenerator(object):
    '''
    RandomGenerator() -> generator object

    usage example:

        >>> piece = TetrisRandomGenerator()
        >>> piece.queue
        ['Z', 'T', 'S', 'I', 'O', 'L', 'J']
        >>> piece.next()
        'J'
        >>> piece.queue
        ['S', 'Z', 'T', 'S', 'I', 'O', 'L']

    Concept:

        Design adapted from: http://tetrisconcept.net/wiki/Random_Generator

        The Random Generator generates the set of all seven tetriminos permuted
        randomly to form a "bag". Tet's from the bag are then fed to a queue.
        When the bag runs out another set of all seven tet's are generated
        and permuted randomly again.
       
   Effect:

        - It is impossible to recieve the same piece more than twice in a row.
        - It's also impossible to recieve a "snake" of 'S' and 'Z'
          tetriminos longer than 4.
        - same tet's will be a maximum of 12 tet's apart, meaning the longest
          you'll have to wait for the next 'I' piece for example, is 12 pieces
          down the queue.

    self.queue[0] -> back of queue
    self.queue[-1] -> front of queue
    '''

    def __init__(self):
        self.all_tets = 'IOTJLSZ'
        self.bag = list(self.all_tets)
        self.queue = list(self.all_tets)
        shuffle(self.bag)
        shuffle(self.queue)

    def __iter__(self):
        return self

    def next(self):
        next_tet = self.queue.pop()
        if not self.bag:
            self.bag = list(self.all_tets)
            shuffle(self.bag)
        self.queue.insert(0, self.bag.pop())
        return next_tet



class TetrisGameEngine(object):
    """
    Design adapted from:
        http://tetrisconcept.net/wiki/Tetris_Guideline

    some word and phrase definitions:

        tetrimino (or just tet)
            - A tetris piece, which is represented by a single
              uppercase character
              "I" | "O" | "T" | "J" | "L" | "Z" | "S"

        mino
            - A single block/cell that is part of a tetrimino form.
            - BTW all tetriminos are made up of exactly 4 minos.

        playing field (or just field)
            - The 10 x 22 grid in which the tetriminoes are moved and placed.
            - The visible field should be 20 rows but the actual playing field
              should be between 22 and 24 rows to allow for a vanish zone (hidden zone).

        bounding box (or just box)
            - One of three different sized grids which contain the form
              of a tetrimino in a certain orientation.
            - This simplifies the effect of the way tetriminoes rotate
              around a point in the SRS system.
            - Bouding box may fall out of playing field but minos may not.

        Super Rotation System (SRS)
            - Used in most modern tetris games according to tetris guidelines
            - In SRS tetriminos rotate around a point and incorporate wall kicks.
            - SRS ultimately allows moves such as the T-spin.

        wall kick
            - essential part of SRS.
            - "O" doesn't rotate so has no wall kick abilities.
            - "I" has it's own set of wall kicks
            - All other tets share a common set of wall kicks.
              wall_kick_data derived from:
                  http://tetrisconcept.net/wiki/SRS

    Schematic/visualisation:

        bbbbbb
        b    b --> bounding box
        bbbbbb
         _ _
        |    |   --> playing field
        |_ _ |

         _ _ _ _ _ _ _ bbbbbbbbbbbbbbbbbbbbb _ _ _ _ _ _ _
        |    |    |    b X  |    |    |    b    |    |    |
        |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |
        |    |    |    b    |    |    |    b    |    |    |
        |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |
        |    |    |    b    |    |    |    b    |    |    |
        |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |_ _ b_ _ |_ _ |_ _ |
        |    |    |    b    |    |    |    b    |    |    |
        |_ _ |_ _ |_ _ bbbbbbbbbbbbbbbbbbbbb_ _ |_ _ |_ _ |
        |    |    |    |    |    |    |    |    |    |    |
        |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |
        |    |    |    |    |    |    |    |    |    |    |
        |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |_ _ |
        |    |    |    |    |    |    |    |    |    |    |

        While the size of the bounding box varies between "I" "O" and the rest,
        tetriminoes are always spawned with the top left corner of their bounding box
        positioned at the cell marked X.

        All tetriminoes are spawned in their first orientation.
        eg. tetriminoes["I"][0] is a bounding box of 4 rows and 4 columns
        containing the I piece like this:
       
        bbbbbbbbbbbbbbbbbbbbb
        b    |    |    |    b
        b_ _ |_ _ |_ _ |_ _ b
        b####|####|####|####b
        b####|####|####|####b
        b    |    |    |    b
        b_ _ |_ _ |_ _ |_ _ b
        b    |    |    |    b
        b    |    |    |    b
        bbbbbbbbbbbbbbbbbbbbb

        "Z", "S", "L", "J", "T" all have a 3 by 3 box
        default/spawn orientation always with piece in contact with
        the top of the box and flat side down.

        bbbbbbbbbbbbbbbb
        b####|####|    b
        b####|####|_ _ b
        b    |####|####b
        b_ _ |####|####b
        b    |    |    b
        b    |    |    b
        bbbbbbbbbbbbbbbb

        "O" bounding box may seem little weird.
        Intuitions says it should just be 2 by 2
        but the 3 by 2 is solely for getting the correct spawn position.

        bbbbbbbbbbbbbbbb
        b    |####|####b
        b_ _ |####|####b
        b    |####|####b
        b_ _ |####|####b
        bbbbbbbbbbbbbbbb
    """

    tetriminoes = {"I":((( '', '', '', ''),
                         ('I','I','I','I'),
                         ( '', '', '', ''),
                         ( '', '', '', '')),

                        (( '', '','I', ''),
                         ( '', '','I', ''),
                         ( '', '','I', ''),
                         ( '', '','I', '')),

                        (( '', '', '', ''),
                         ( '', '', '', ''),
                         ('I','I','I','I'),
                         ( '', '', '', '')),

                        (( '','I', '', ''),
                         ( '','I', '', ''),
                         ( '','I', '', ''),
                         ( '','I', '', ''))),

                   "O":((( '','O','O'),
                         ( '','O','O')),),

                   "T":((( '','T', ''),
                         ('T','T','T'),
                         ( '', '', '')),

                        (( '','T', ''),
                         ( '','T','T'),
                         ( '','T', '')),

                        (( '', '', ''),
                         ('T','T','T'),
                         ( '','T', '')),

                        (( '','T', ''),
                         ('T','T', ''),
                         ( '','T', ''))),

                   "J":((('J', '', ''),
                         ('J','J','J'),
                         ( '', '', '')),

                        (( '','J','J'),
                         ( '','J', ''),
                         ( '','J', '')),

                        (( '', '', ''),
                         ('J','J','J'),
                         ( '', '','J')),

                        (( '','J', ''),
                         ( '','J', ''),
                         ('J','J', ''))),

                   "L":((( '', '','L'),
                         ('L','L','L'),
                         ( '', '', '')),

                        (( '','L', ''),
                         ( '','L', ''),
                         ( '','L','L')),

                        (( '', '', ''),
                         ('L','L','L'),
                         ('L', '', '')),

                        (('L','L', ''),
                         ( '','L', ''),
                         ( '','L', ''))),

                   "S":((( '','S','S'),
                         ('S','S', ''),
                         ( '', '', '')),

                        (( '','S', ''),
                         ( '','S','S'),
                         ( '', '','S')),

                        (( '', '', ''),
                         ( '','S','S'),
                         ('S','S', '')),

                        (('S', '', ''),
                         ('S','S', ''),
                         ( '','S', ''))),

                   "Z":((('Z','Z', ''),
                         ( '','Z','Z'),
                         ( '', '', '')),

                        (( '', '','Z'),
                         ( '','Z','Z'),
                         ( '','Z', '')),

                        (( '', '', ''),
                         ('Z','Z', ''),
                         ( '','Z','Z')),

                        (( '','Z', ''),
                         ('Z','Z', ''),
                         ('Z', '', '')))}

    # http://tetrisconcept.net/wiki/SRS
    # wall_kick_data holds (x, y) kick values found on page from link above.
    # only differences are how the keys are represented
    # In the original data '0->L' is the key for rotating left from orientation 0.
    # Here it become '03' for rotating from orientation 0 to orientation 3 which
    # translates to the same thing.
    # Also all y values were negated (changed to opposite sign)
    # because in the code here "up" means -y not +y.
    # This is simply because the cartesian planes used in math is upside down when
    # veiwed from the perspective of nested lists/arrays in most programming languages.
    wall_kick_data = {"I":{"01":(( 0, 0), (-2, 0), ( 1, 0), (-2, 1), ( 1,-2)),
                           "10":(( 0, 0), ( 2, 0), (-1, 0), ( 2,-1), (-1, 2)),
                           "12":(( 0, 0), (-1, 0), ( 2, 0), (-1,-2), ( 2, 1)),
                           "21":(( 0, 0), ( 1, 0), (-2, 0), ( 1, 2), (-2,-1)),
                           "23":(( 0, 0), ( 2, 0), (-1, 0), ( 2,-1), (-1, 2)),
                           "32":(( 0, 0), (-2, 0), ( 1, 0), (-2, 1), ( 1,-2)),
                           "30":(( 0, 0), ( 1, 0), (-2, 0), ( 1, 2), (-2,-1)),
                           "03":(( 0, 0), (-1, 0), ( 2, 0), (-1,-2), ( 2, 1))},

                      "T":{"01":((0, 0), (-1, 0), (-1,-1), (0, 2), (-1, 2)),
                           "10":((0, 0), ( 1, 0), ( 1, 1), (0,-2), ( 1,-2)),
                           "12":((0, 0), ( 1, 0), ( 1, 1), (0,-2), ( 1,-2)),
                           "21":((0, 0), (-1, 0), (-1,-1), (0, 2), (-1, 2)),
                           "23":((0, 0), ( 1, 0), ( 1,-1), (0, 2), ( 1, 2)),
                           "32":((0, 0), (-1, 0), (-1, 1), (0,-2), (-1,-2)),
                           "30":((0, 0), (-1, 0), (-1, 1), (0,-2), (-1,-2)),
                           "03":((0, 0), ( 1, 0), ( 1,-1), (0, 2), ( 1, 2))},
                      }
    for other in "LJSZ":
        wall_kick_data[other] = wall_kick_data["T"]

    # score = scoring[line_clears]
    scoring = [0, 100, 300, 500, 800]

    def __init__(self, width=10, height=22):
        self.playing_field = [["" for x in range(width)]
                              for y in range(height)]
        self.field_width = width
        self.field_height = height
        self.box = None
        # box will contain a value from tetriminoes variable.
        # box grid is seperate from playing field grid.
        # Use field_state() method to get a single grid
        # with playing field and box combined.

        self.box_coords = None
        # box_coords will contain (x, y) of the cell at box[0][0].
        # Spawn coords for any box is (3, 0).

        self.tet_state = None
        # tet_state describes what is in the bounding box.
        # eg. ["I", 0] for tetrimino "I" at orientation 0.
        # tet_state = None when no box is in the field.

        self.line_clears = 0
        self.score = 0

    def get_field_state(self):
        """
        Returns copy of playing_field with box pasted in.
        Returns False when game over.
        """
        field = [list(row) for row in self.playing_field] # copy
        if self.box_coords == None:
            return field
        else:
            x, y = self.box_coords
            return self.__merge_box_and_field(self.box, field, x, y)

    def spawn(self, tet):
        """
        Places tet in the playing field. tet -> str
        """
        self.box = self.tetriminoes[tet][0]
        self.box_coords = (3, 0)
        self.tet_state = [tet, 0]

    def move(self, direction):
        """
        Changes coords of current tet. Returns True if successful.
        direction -> "L" | "R" | "D"
        """
        x, y = self.box_coords
        if direction == "L":
            x -= 1
        elif direction == "R":
            x += 1
        elif direction == "D":
            y += 1
        else:
            raise ValueError
        if self.__merge_box_and_field(self.box, self.playing_field, x, y, simulate=True):
            self.box_coords = (x, y)
            return True
        else:
            return False

    def hard_drop(self):
        """
        Hard drops current tetrimino. Returns nothing.
        """
        if self.box:
            while self.move("D"):
                pass
        self.playing_field = self.__merge_box_and_field(self.box, self.playing_field,
                                                        *self.box_coords, simulate=False)
        self.box = None
        self.box_coords = None
        self.tet_state = None

    def rotate(self, direction):
        """
        Rotates current tetrimino. Returns True if successful.
        direction -> "L" | "R"
        """
        tet, r_start = self.tet_state
        if tet == "O":
            return False
        r_end = r_start
        r_end += (1 if direction == "R" else -1)
        r_end %= 4
        kick_key = str(r_start) + str(r_end)
        wall_kicks = self.wall_kick_data[tet][kick_key]
        box = self.tetriminoes[tet][r_end]
        field = self.playing_field
        x_start, y_start = self.box_coords
        for kick in wall_kicks:
            x_end = x_start + kick[0]
            y_end = y_start + kick[1]
            if self.__merge_box_and_field(box, field, x_end, y_end, simulate=True):
                self.box = box
                self.box_coords = (x_end, y_end)
                self.tet_state[1] = r_end
                return True
        return False

    def clear_lines(self):
        """
        Ammends playing_field, line_clears, score and returns score (not total).
        """
        clears = 0
        field = self.playing_field
        for y in xrange(self.field_height):
            if all(map(lambda x: x != '', field[y])):
                del(field[y])
                field.insert(0, ["" for x in range(self.field_width)])
                clears += 1
        if clears:
            self.line_clears += clears
            score = self.scoring[clears]
            self.score += score
            return score

    def __merge_box_and_field(self, box, field, x, y, simulate=False):
        """
        simulate=False -> returns field with box pasted in.
        simulate=True -> returns True if no collisions.
        """
        for box_y in xrange(len(box)):
            for box_x in xrange(len(box[0])):
                field_x = x + box_x
                field_y = y + box_y
                if ((field_x < 0 or field_x >= self.field_width) or
                    (field_y < 0 or field_y >= self.field_height)):
                    # box out of playing field
                    if box[box_y][box_x]:
                        # mino out of playing field
                        return False
                    else:
                        continue
                elif field[field_y][field_x] and box[box_y][box_x]:
                    # collision
                    return False
                elif not simulate:
                    field[field_y][field_x] += box[box_y][box_x]
                else:
                    pass

        if simulate:
            return True
        else:
            return field


## --------- simple testing ---------->
from random import choice
from os import system
from time import sleep

def display():
   "trivial function to see what's going on"
   field = GAME.get_field_state()
   print GAME.score
   print GAME.line_clears
   sleep(0.01)
   system('clear')
   print '\n'.join([''.join(['[#]' if cell else '[_]' for cell in row])
                    for row in field])

def randmove():
   "performs a random rotation or movement"
   if choice((0, 1)):
       GAME.move(choice(['D', 'L', 'R']))
   else:
       GAME.rotate(choice(['L', 'R']))

PIECE = TetrisRandomGenerator()
GAME = TetrisGameEngine(width = 6)

for tet in range(100):
   GAME.spawn(PIECE.next())
   display()
   for movement in range(5):
       randmove()
       display()
   GAME.hard_drop()
   display()