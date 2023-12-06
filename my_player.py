from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time
import numpy as np
from board_abalone import BoardAbalone
import random


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(
        self, piece_type: str, name: str = "bob", time_limit: float = 60 * 15, *args
    ) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)
        self.zobrist_keys = {}
        self.first_action = True
        self.visited_positions = []
        self.visitedStatesCounter = 0

    def init_zobrist(self, board: BoardAbalone, bitstring_length=64):
        """
        Initialize the zobrist keys that will be use for hashing

        Args:
            board: BoardAbalone, a representation of the game
            bitstring_length: int, the length of random bitstrings use for keys
        """
        board_array = board.get_grid()
        for row in range(len(board_array)):
            for column in range(len(board_array[0])):
                for piece in [1, 2]:
                    # Possible tile states are:
                    # 0: unaccessible tile
                    # 1: my pieces
                    # 2: opponent piece
                    # 3: empty tile
                    # But only tiles with pieces interest us
                    self.zobrist_keys[
                        "{},{},{}".format(row, column, piece)
                    ] = random.getrandbits(bitstring_length)
        return None

    def zobrist_hash(self, state: GameState):
        """
        Computes the zobrist hash for the current state

        Args:
            board: BoardAbalone, the representation of the current game state

        Returns:
            h: int, zobrist hash of the state
        """
        h = 0
        board = state.get_rep().get_grid()
        for row in range(len(board)):
            for column in range(len(board[0])):
                piece = board[row][column]
                if piece == "W" or piece == "B":
                    piece_type = 1 if piece == "W" else 2
                    h = h ^ self.zobrist_keys.get(
                        ("{},{},{}".format(row, column, piece_type)), None
                    )
        return h

    def heuristique(self, state: GameState):
        """
        Computes the heuristic of the given state
        The heuristic takes into account:
            - The opponent's score
            - The proximity of our pieces
            - The proximity to the edge of our pieces
            - The proximity of the opponent's pieces
            - The proximity of the opponent's pieces to the edge

        Args:
            state: GameState , the state on which to apply the heuristic
        """
        """
        Techniques d'allocations du temps de recherche
        """

        score = 0
        current_rep = state.get_rep()
        b = current_rep.get_env()
        distance_my_player = 0
        distance_other_player = 0
        for i, j in list(b.keys()):
            p = b.get((i, j), None)
            if p.get_owner_id() == self.id:
                distance_my_player += abs(8 - i) + abs(4 - j)
            else:
                distance_other_player += abs(8 - i) + abs(4 - j)
        distance_total = distance_my_player + distance_other_player
        score += 0.5 * (distance_other_player - distance_my_player) / distance_total

        grid = state.get_rep().get_grid()
        n = len(grid)
        n_links_my_player = 0
        n_links_other_player = 0
        type = self.piece_type
        if type == "W":
            other_type = "B"
        else:
            other_type = "W"
        for i in range(n - 1):
            for j in range(n - 1):
                # On regarde sur les lignes
                if grid[i][j] == type and grid[i][j + 1] == type:
                    n_links_my_player += 1
                if grid[i][j] == other_type and grid[i][j + 1] == other_type:
                    n_links_other_player += 1

                # On regarde sur les colonnes
                if grid[i][j] == type and grid[i + 1][j] == type:
                    n_links_my_player += 1
                if grid[i][j] == other_type and grid[i + 1][j] == other_type:
                    n_links_other_player += 1

                # On regarde sur les diagonal S-E
                if grid[i][j] == type and grid[i + 1][j + 1] == type:
                    n_links_my_player += 1
                if grid[i][j] == other_type and grid[i + 1][j + 1] == other_type:
                    n_links_other_player += 1

                # On regarde sur les diagonal S-O
                if j >= 1 and grid[i][j] == type and grid[i + 1][j - 1] == type:
                    n_links_my_player += 1
                if (
                    j >= 1
                    and grid[i][j] == other_type
                    and grid[i + 1][j - 1] == other_type
                ):
                    n_links_other_player += 1
        score += (
            0.25
            * (n_links_my_player - n_links_other_player)
            / (n_links_my_player + n_links_other_player)
        )

        for key in state.scores.keys():
            if key == self.id:
                score += state.scores[key] / 2
            else:
                score -= state.scores[key]
        return score

    def maxValueWithTranspo(self, state: GameState, alpha, beta, theta=0, limite=40000):
        if state.is_done():
            return (0, None, theta)

        v_star = -np.inf
        m = None
        possible_actions = list(state.get_possible_actions())
        for action in possible_actions:
            next_state = action.get_next_game_state()
            next_state_hash = self.zobrist_hash(next_state)
            if theta == limite:
                v = self.heuristique(next_state)
            else:
                theta += 1
                if not (next_state_hash in self.visited_positions):
                    self.visited_positions.append(next_state_hash)
                    v, _, theta = self.minValueWithTranspo(
                        next_state, alpha, beta, theta
                    )
                    if v > v_star:
                        v_star = v
                        m = action
                        alpha = max(alpha, v_star)

        if v_star >= beta:
            return (v_star, m, theta)
        return (v_star, m, theta)

    def minValueWithTranspo(self, state: GameState, alpha, beta, theta=0, limite=40000):
        if state.is_done():
            return (-6, None, theta)
        v_star = np.inf
        m = None
        possible_actions = list(state.get_possible_actions())
        for action in possible_actions:
            next_state = action.get_next_game_state()
            next_state_hash = self.zobrist_hash(next_state)
            if theta == limite:
                v = self.heuristique(next_state)
            else:
                theta += 1
                if not (next_state_hash in self.visited_positions):
                    self.visited_positions.append(next_state_hash)
                    v, _, theta = self.maxValueWithTranspo(
                        next_state, alpha, beta, theta
                    )
                    if v < v_star:
                        v_star = v
                        m = action
                        beta = min(beta, v_star)
        if v_star <= alpha:
            return (v_star, m, theta)
        return (v_star, m, theta)

    def maxValue(self, state: GameState, alpha, beta, theta=0, limite=40000):
        if state.is_done():
            return (0, None, theta)
        v_star = -np.inf
        m = None
        possible_actions = list(state.get_possible_actions())
        for action in possible_actions:
            next_state = action.get_next_game_state()
            if theta == limite:
                v = self.heuristique(next_state)
            else:
                theta += 1
                v, _, theta = self.minValue(next_state, alpha, beta, theta)
            if v > v_star:
                v_star = v
                m = action
                alpha = max(alpha, v_star)
        if v_star >= beta:
            return (v_star, m, theta)
        return (v_star, m, theta)

    def minValue(self, state: GameState, alpha, beta, theta=0, limite=40000):
        if state.is_done():
            return (-6, None, theta)
        v_star = np.inf
        m = None
        possible_actions = list(state.get_possible_actions())
        for action in possible_actions:
            next_state = action.get_next_game_state()
            if theta == limite:
                v = self.heuristique(next_state)
            else:
                theta += 1
                v, _, theta = self.maxValue(next_state, alpha, beta, theta)
            if v < v_star:
                v_star = v
                m = action
                beta = min(beta, v_star)
        if v_star <= alpha:
            return (v_star, m, theta)
        return (v_star, m, theta)

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
        if self.first_action:
            self.init_zobrist(current_state.rep)

        if TRANSPO:
            _, action, _ = self.maxValueWithTranspo(current_state, -6, 0)
        else:
            self.visitedStatesCounter = 0
            _, action, _ = self.maxValue(current_state, -6, 0)

        self.first_action = False
        self.visited_positions = []
        return action


TRANSPO = False
