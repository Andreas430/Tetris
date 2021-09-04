#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!/usr/bin/env python
# coding: utf-8

# In[85]:



import random
import cv2
import numpy as np
from PIL import Image
from time import sleep
import matplotlib.pyplot as plt


# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    moves = []
    for i in range(6):
        for j in range(0,360,90):
            moves.append((i,j))

    action_map = {i:move for i,move in enumerate(moves)}
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 6
    BOARD_HEIGHT = 8

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0)],
            90: [(0,2), (0,1), (0,0)],
            180: [(2,1), (1,1), (0,1)],
            270: [(1,0), (1,1), (1,2)],
        },
        1: { # L
            0: [(0,0), (1,0), (1,1)],
            90: [(0,1), (0,0), (1,0)],
            180: [(0,0), (0,1), (1,1)],
            270: [(0,1), (1,1), (1,0)],
        },

    }

    COLORS = {
        0: (255, 255, 255),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0,0,0),
        4: (0,128,0),
    }


    def __init__(self, mode, render_mode = 'normal'):
        self.mode = mode
        self.render_mode = render_mode
        if self.render_mode == 'normal': self.testing = Tetris.BOARD_WIDTH
        else: self.testing = Tetris.BOARD_WIDTH +5
        
        self.reset()

    
    def reset(self):
        '''Resets the game, returning the current state'''
        
        self.board = [[0] * (Tetris.BOARD_WIDTH) for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag[-1]
        self._new_round()
        self.score = 0
        self.lines_cleared = 0
            
        
        if self.mode == 'glimpse':
            return np.array(self.glimpse())
        elif self.mode == 'board':
            return np.array(self.board)
        elif self.mode == '4feat':
            return np.array(self._get_4board_props(self.board))
        elif self.mode == '12feat':
            return np.array(self._get_12board_props(self.board))


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.
        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        
        self.current_piece = self.next_piece
        random.shuffle(self.bag)
        self.next_piece = self.bag[-1]
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''

        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH                     or y < 0 or y >= Tetris.BOARD_HEIGHT                     or self.board[y][x] == Tetris.MAP_BLOCK:
                
                return True
       
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_4board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def _get_12board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        h1, h2 ,h3, h4, h5, h6 = np.sum(board, axis = 0)
        return [lines, holes, total_bumpiness, sum_height, h1 ,h2 ,h3, h4, h5, h6, self.current_piece, self.next_piece]



    def play(self, move, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        action = Tetris.action_map[move]
        x = action[0]
        rotation =action[1]
        if self.current_piece == 0:
            if rotation in [0,180]:
                rotation = 0
                if x > 3:
                    x =3
            else:
                rotation = 90
        else:
            if x >4:
                x = 4
            
        self.current_pos = [x, 0]
        self.current_rotation = rotation
        # Drop piece
        self.copy_board = self.board.copy()
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
            
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        self.lines_cleared, self.board = self._clear_lines(self.board)
#         score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH

        reward = 0
        
        # Start new round
        self._new_round()
        if self.game_over:
            reward = -10
        elif self.lines_cleared == 1:
            reward = 1
        elif self.lines_cleared == 2:
            reward = 4
        elif self.lines_cleared == 3:
            reward = 16
        
        if self.mode == 'glimpse':
            return np.array(self.glimpse()), reward, self.game_over
        elif self.mode == 'board':
            return np.array(self.glimpse()), reward, self.game_over
        elif self.mode == '4feat':
            return np.array(self._get_4board_props(self.board)), reward, self.game_over
        elif self.mode == '12feat':
            return np.array(self._get_12board_props(self.board)), reward, self.game_over
        
        
    

    def glimpse(self):
        new_board = [[3]+[0]*4 for i in range(8)]

        piece = [(x+3,y+2) for x,y in Tetris.TETROMINOS[self.next_piece][0]]
        for x,y in piece:
            new_board[x][y] = 4
        random_bo = self._get_complete_board()
        result = [random_bo[i] + new_board[i] for i in range(8)]
        return result
    

    def render(self):
        '''Renders the current board'''
        
        if self.render_mode != 'normal': 
            new_board = [[3]+[0]*4 for i in range(8)]

            piece = [(x+3,y+2) for x,y in Tetris.TETROMINOS[self.next_piece][0]]
            for x,y in piece:
                new_board[x][y] = 4
            random_bo = self._get_complete_board()
            result = [random_bo[i] + new_board[i] for i in range(8)]
        else:
            result = self._get_complete_board()

        img = [Tetris.COLORS[p] for row in result for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, self.testing, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
#         img = img.resize(((self.testing) * 25,Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        img=cv2.resize(img,(self.testing *50, Tetris.BOARD_HEIGHT * 50),interpolation = cv2.INTER_AREA)
#         cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)


# In[ ]:




