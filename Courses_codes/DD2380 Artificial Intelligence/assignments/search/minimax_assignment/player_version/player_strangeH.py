#!/usr/bin/env python3
import random
from math import inf
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import timeit
from time import time
import logging
import inspect
from copy import deepcopy, copy



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
        self.action=0
        self.hash=dict()
        self.start_time = 0
        self.abort_before_timeout = True


    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()
        ##print(first_msg)
        while True:
            msg = self.receiver()
            # Create the root node of the game tree
            node = Node(message=msg, player=0)
            # Possible next moves: "stay", "left", "right", "up", "down"
            self.start_time = time()
            self.abort_before_timeout = True
            self.hash=dict()
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    

    def search_best_next_move(self, initial_tree_node):
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
        depthmax=8
        alpha=-1e5
        beta=1e5
        self.action=0
        initial_tree_node.heuristic_score = self.heuristic_1(initial_tree_node)
        self.alphabeta(initial_tree_node,depthmax,alpha,beta)

        #print('action')
        #print(self.action)
        return ACTION_TO_STR[self.action]
    def hash_state(self,state):
        return hash(
            (
                hash(frozenset(state.hook_positions.items())),
                hash(frozenset(state.fish_positions.items())),
            )
        )

    def l1_cylinder(self,hook, other_hook, fish):
        # TODO: consider the other boat as the origin of X
        normalized_fish_x = (fish[0] - other_hook[0]) % 20
        normalized_hook_x = (hook[0] - other_hook[0]) % 20
        y = abs(fish[1] - hook[1])
        x = abs(normalized_fish_x - normalized_hook_x)
        # x = min(delta_x, 20 - delta_x)

        l1 = x + y
        # logging.info(f"l1={l1}, x={x}, y={y}, h={hook}, f={fish}, oh={other_hook}")
        return l1 

    def heuristic_1(self,node, fish_score_scaler=0.5):
        
        s= node.state

        hooks_pos = deepcopy(s.hook_positions)
        fish_scores = s.fish_scores

        v = 0
        caught = 0

        for fish, pos in s.fish_positions.items():
            fish_score = fish_scores[fish]
            d_0 = self.l1_cylinder(hooks_pos[0], hooks_pos[1], pos)

            if fish_score < 0:
                d_0 = max(-1,-d_0)
                fish_score = -1

            d_1 = self.l1_cylinder(hooks_pos[1], hooks_pos[0], pos)

            if len(s.fish_positions) == 1:
                print(len(s.fish_positions))
                if d_0 == 0:
                    caught += fish_score

                if d_1 == 0:
                    caught -= fish_score

            else:
                caught=0

            f = d_1 * 2 - d_0 * 0.5 + pos[1] * 1 + fish_score_scaler * fish_score 
            v += f * 0.01

        g = s.player_scores[0] - s.player_scores[1]
        heuristic_score = 0.1 * v + g + caught
        return heuristic_score 


    def alphabeta(self,node,depth,alpha,beta):
        #print('depth')
        #print(depth)
        
        hashed_state = self.hash_state(node.state)
        if hashed_state in self.hash:
            return self.hash[hashed_state]
        time_left = 75e-3 - (time() - self.start_time)
        children = node.compute_and_get_children()
        if (len(children) == 1 and children[0].move == 1) or depth==0 or len(children) == 0 or time_left < 0.023:#(self.time_buffer + 0.002*1.8**depth):
            self.hash[hashed_state] = node.heuristic_score #score 
            return node.heuristic_score
    
        for child in children:
            child.heuristic_score = self.heuristic_1(child)
        if node.state.player==0:
            v=-1e5
            children = list(sorted(children, key=lambda c: c.heuristic_score, reverse=True))
            for child in children:
                temp=alpha
                v=max(v,self.alphabeta(child,depth-1,alpha,beta))
                           
                alpha=max(alpha,v)
                if depth==8 and alpha!=temp:
                    self.action=child.move                      
                if beta<=alpha:
                    break
        
        else:
            v=1e5
            children = list(sorted(children, key=lambda c: c.heuristic_score, reverse=False))
            for child in children:
                v=min(v,self.alphabeta(child,depth-1,alpha,beta))
                beta=min(beta,v)
                if beta<=alpha:
                    break
        
        return v
