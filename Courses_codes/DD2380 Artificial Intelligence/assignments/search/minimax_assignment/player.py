#!/usr/bin/env python3
import numpy as np
from time import time
from typing import Optional

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return

class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """
        # Initialize your minimax model
        first_msg = self.receiver()
        model = self.initialize_model()

        while True:
            msg = self.receiver()
            # Create the root node of the game tree
            node = Node(message=msg, player=0)
            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move, s_time = self.search_best_next_move(model=model, initial_tree_node=node)
            # Execute next action
            self.sender({"action": best_move, "search_time": s_time})
    
    def initialize_model(self):
        """
        Initialize your minimax model 
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3}, 
          'fish1': {'score': 2, 'type': 1}, 
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return Minimax_Search()  


    def search_best_next_move(self, model, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        
        #try to create the search tree
        
        # if player is already caught a fish, action is only up
        if initial_tree_node.state.get_caught()[0] != None:
            action = 1
            search_time = 0 
        else:
            model.set_root(Search_tree_node(initial_tree_node))
            model.set_depth(12)
            action, search_time = model.find_best_action()
        return ACTION_TO_STR[action], search_time
    
class Search_tree_node():
    '''
    This class for sorted the children of a node 
    by recording extra hvalue and mvalue
    '''
    def __init__(self, node: Node):
        # save extra heuristic value and minimax value of each node
        self.hvalue = None
        self.mvalue = None
        self.node = node
        self.children = None
    
    def set_and_get_hvalue(self):
        self.hvalue = heuristic_value2(self.node) #111111111111111
        return self.hvalue

    def set_mvalue(self, value):
        self.mvalue = value

    def compute_and_get_sorted_children(self):
        if self.children:
            return self.children
        
        self.children = []
        flag = False
        if self.node.state.player == 0:
            flag = True
        
        for child in self.node.compute_and_get_children():
            Search_tree_child = Search_tree_node(child)
            Search_tree_child.set_and_get_hvalue()
            self.children.append(Search_tree_child)
        
        self.children.sort(key = lambda v: v.mvalue or v.hvalue, reverse = flag)

        return self.children

class Minimax_Search():
    def __init__(self):
        self.root = None
        self.depth = None
        self.max_runtime = 75e-3 - 18e-3 # 25e-3 is time for functions return
        self.search_start_time = None
    
    def set_root(self, node: Search_tree_node):
        self.root = node
    
    def set_depth(self, depth):
        self.depth = depth
    
    def find_best_action(self):
        self.search_start_time = time()
        children =  self.root.compute_and_get_sorted_children()
        best_child = None
        best_value = -np.inf
        alpha = -np.inf

        for depth in range(self.depth):
            #print(depth)
            value = -np.inf
            # the player in root is max, so the below code is a part of alphabeta_search
            for child in children:
                if time() - self.search_start_time >= self.max_runtime:
                    depth = 1

                value = max(value, self.alphabeta_search(child, depth, alpha, np.inf))
                alpha = max(alpha, value)

                # update best_value and best_child
                if value > best_value:
                    best_value = value
                    best_child = child

            # exit the search if time is not enough
            if time() - self.search_start_time >= self.max_runtime:
                break

        return best_child.node.move, time() - self.search_start_time


    def alphabeta_search(self, STnode: Search_tree_node, depth, alpha, beta):
        if depth == 0:
            return STnode.set_and_get_hvalue()
        else:
            children = STnode.compute_and_get_sorted_children()
            if STnode.node.state.player == 0:
                v = -np.inf
                for child in children:
                    if time() - self.search_start_time >= self.max_runtime:
                        depth = 1
                    v = max(v, self.alphabeta_search(child, depth - 1, alpha, beta))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                STnode.set_mvalue(v)
            else:
                v = np.inf
                for child in children:
                    if time() - self.search_start_time >= self.max_runtime:
                        depth = 1
                    v = min(v, self.alphabeta_search(child, depth - 1, alpha, beta))
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
                STnode.set_mvalue(v)
        return v 

def heuristic_value(node, fish_score_scale = 10):
    '''
    return the heuristic value of the state of the input node
    '''
    # if node.state is already in the self.hash = dict(), then directly return the H value
    state = node.state

    # h(state) = curr_player_score + K * fish_score(if fish get caught) + distance_score
    
    #compute curr_player_score
    curr_player1_score, curr_player2_score  = state.get_player_scores()
    curr_player_score = curr_player1_score - curr_player2_score
    
    # compute fish_score(if fish get caught)
    fish_score1 = 0
    fish_score2 = 0
    fish_on_rod1, fish_on_rod2 = state.get_caught()
    if fish_on_rod1:
        fish_score1 = state.get_fish_scores()[fish_on_rod1] 
    if fish_on_rod2:
        fish_score2 = state.get_fish_scores()[fish_on_rod2]
    fish_score = fish_score1 - fish_score2 

    # compute distance_score
    hooks = state.get_hook_positions()
    x1, y1 = hooks[0]
    x2, y2 = hooks[1]
    distance_score1 = 0
    distance_score2 = 0
    for fish, fishposition in state.get_fish_positions().items():
        # rule out the fish if it got caught
        if fish == fish_on_rod1 or fish == fish_on_rod2:
            continue

        xf, yf = fishposition
        
        # min(abs(x1 - xf), 20 - abs(x1 - xf)) whether to go across the boundary
        distance1 = min(abs(x1 - xf), 20 - abs(x1 - xf)) + abs(y1 - yf)
        distance2 = min(abs(x2 - xf), 20 - abs(x2 - xf)) + abs(y2 - yf)
        distance_score1 += 1 / (distance1 + 1) * state.get_fish_scores()[fish] + yf
        distance_score2 += 1 / (distance2 + 1) * state.get_fish_scores()[fish] 
    
    distance_score = distance_score1 - distance_score2

    score = curr_player_score + fish_score_scale * fish_score +  0.001 * distance_score
    # self.hash[key] = score
    return score

def distance(hook,fish):
        y = abs(fish[1] - hook[1])
        x = min(20-abs(fish[0] - hook[0]),abs(fish[0] - hook[0]))
        distance = x + y
        return distance 

def heuristic_value2(node):
    hooks_pos = node.state.hook_positions
    fish_scores = node.state.fish_scores
    v = 0
    caught = node.state.get_caught()[0]
    if caught :
        v += 0.5*fish_scores[caught]
    for fish, pos in node.state.fish_positions.items():
        this_fish = fish_scores[fish]
        d_0 = distance(hooks_pos[0],pos)
        d_1 = distance(hooks_pos[1],pos)
        if this_fish < 0:
            d_0 = max(-1,-d_0)
            d_1 = max(-1,-d_1)
            this_fish = max(-1,this_fish)
        f = 4*d_1 - 2*d_0 + pos[1] + this_fish 
        v += f 
    g = node.state.player_scores[0] - node.state.player_scores[1]
    heuristic_score =0.0001*v + g 
    return heuristic_score 

def heuristic_value3(node, fish_score_scale = 10):
    '''
    return the heuristic value of the state of the input node
    '''
    # if node.state is already in the self.hash = dict(), then directly return the H value
    state = node.state

    # h(state) = curr_player_score + K * fish_score(if fish get caught) + distance_score
    
    #compute curr_player_score
    curr_player1_score, curr_player2_score  = state.get_player_scores()
    curr_player_score = curr_player1_score - curr_player2_score
    
    # compute fish_score(if fish get caught)
    fish_score1 = 0
    fish_score2 = 0
    fish_on_rod1, fish_on_rod2 = state.get_caught()
    if fish_on_rod1:
        fish_score1 = state.get_fish_scores()[fish_on_rod1] 
    if fish_on_rod2:
        fish_score2 = state.get_fish_scores()[fish_on_rod2]
    fish_score = fish_score1 - fish_score2 

    # compute distance_score
    hooks = state.get_hook_positions()
    x1, y1 = hooks[0]
    x2, y2 = hooks[1]
    distance_score1 = 0
    distance_score2 = 0
    for fish, fishposition in state.get_fish_positions().items():
        # rule out the fish if it got caught
        if fish == fish_on_rod1 or fish == fish_on_rod2:
            continue

        xf, yf = fishposition
        
        # min(abs(x1 - xf), 20 - abs(x1 - xf)) whether to go across the boundary
        distance1x = min(abs(x1 - xf), 20 - abs(x1 - xf)) 
        distance1y = abs(y1 - yf)
        distance2x = min(abs(x2 - xf), 20 - abs(x2 - xf))
        distance2y = abs(y2 - yf)
        if state.get_fish_scores()[fish] < 0:
            distance_score1 = 2 * (1 / (distance1x + 1) + 1 / (distance1y + 1))*state.get_fish_scores()[fish]
        else:
            distance_score1 += (1 / (distance1x + 1) + 2 / (distance1y + 1))*state.get_fish_scores()[fish] + yf
            distance_score2 += (1 / (distance2x + 1) + 2 / (distance2y + 1))*state.get_fish_scores()[fish] 
    
    distance_score = distance_score1 - distance_score2

    score = fish_score_scale * fish_score + 0.001 * distance_score + 1 * curr_player_score
    # self.hash[key] = score
    return score
       
   
    