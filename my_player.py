from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time
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
        temps_debut = time.time()
        def heuristique(state_to_evaluate):
            """
            - prendre en compte le score adverse
            - prendre en compte le fait que nos boules soient resserées
            - prendre en compte que celles de l'adversaire soient disséminées
            - prendre en compte que les boules de l'adversaire soient proches du bord
            - prendre en compte du fait que nos boules sont proches du centre
            """
            """
            Techniques d'allocations du temps de recherche
            """
            
            score = 0
            current_rep = state_to_evaluate.get_rep()
            b = current_rep.get_env()
            distance_my_player = 0
            distance_other_player = 0
            for i, j in list(b.keys()):
                p = b.get((i, j), None)
                if p.get_owner_id() == self.id:
                    distance_my_player += abs(8-i) + abs(4-j)
                else: 
                    distance_other_player += abs(8-i) + abs(4-j)
            distance_total = distance_my_player + distance_other_player
            score += 0.5*(distance_other_player - distance_my_player)/distance_total 

            grid = state_to_evaluate.get_rep().get_grid()
            n = len(grid)
            n_links_my_player = 0
            n_links_other_player = 0
            type = self.piece_type
            if type == "W":
                other_type = "B"
            else:
                other_type = "W"
            for i in range(n-1):
                for j in range(n-1):
                    # On regarde sur les lignes
                    if grid[i][j] == type and grid[i][j+1] == type :
                        n_links_my_player += 1
                    if grid[i][j] == other_type and grid[i][j+1] == other_type :
                        n_links_other_player += 1 

                    # On regarde sur les colonnes
                    if grid[i][j] == type and grid[i+1][j] == type :
                        n_links_my_player += 1
                    if grid[i][j] == other_type and grid[i+1][j] == other_type :
                        n_links_other_player += 1 

                    # On regarde sur les diagonal S-E
                    if grid[i][j] == type and grid[i+1][j+1] == type :
                        n_links_my_player += 1
                    if grid[i][j] == other_type and grid[i+1][j+1] == other_type :
                        n_links_other_player += 1 
                    
                    # On regarde sur les diagonal S-O
                    if j>= 1 and grid[i][j] == type and grid[i+1][j-1] == type :
                        n_links_my_player += 1
                    if j>= 1 and grid[i][j] == other_type and grid[i+1][j-1] == other_type :
                        n_links_other_player += 1 
            score += 0.25*(n_links_my_player - n_links_other_player)/(n_links_my_player + n_links_other_player)

            for key in state_to_evaluate.scores.keys():
                if key == self.id:
                    score += state_to_evaluate.scores[key]/2
                else:
                    score -=  state_to_evaluate.scores[key] 
            return score 
        
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
        
        limite = 40000
        _, action, _ = maxValue(current_state,- 6, 0)
        return action
