from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import random
import numpy as np


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type,name,time_limit,*args)


    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """

        def heuristique(state_to_evaluate):
            return state_to_evaluate.compute_scores(id_add=self.id)[self.id]
        
        
        def maxValue(state,alpha,beta,theta = 0):
            if state.is_done():
                return (0,None,theta)
            v_star = - np.inf
            m = None
            possible_actions = list(state.get_possible_actions())
            for action in possible_actions:
                next_state = action.get_next_game_state()
                if theta == limite : 
                    v = heuristique(next_state)
                else :
                    theta +=1
                    v,_,theta = minValue(next_state,alpha,beta, theta)
                if v > v_star:
                    v_star = v
                    m = action
                    alpha = max(alpha,v_star)
            if v_star >= beta:
                return(v_star, m, theta)
            return(v_star, m, theta)
        
        def minValue(state, alpha, beta, theta = 0):
            if state.is_done():
                return (-6, None,theta)
            v_star = np.inf
            m = None
            possible_actions = list(state.get_possible_actions())
            for action in possible_actions:
                next_state = action.get_next_game_state()
                if theta == limite:
                    v = heuristique(next_state)
                else:
                    theta += 1
                    v,_,theta = maxValue(next_state,alpha,beta,theta)
                if v < v_star:
                    v_star = v
                    m = action
                    beta = min(beta,v_star)
            if v_star <= alpha:
                    return(v_star, m, theta)
            return(v_star, m,theta)
        
        limite = 10000
        _, action, _ = maxValue(current_state,- 6, 0)
        return action
