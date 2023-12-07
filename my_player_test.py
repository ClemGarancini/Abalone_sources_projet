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
        self.maxDepth = 10
        self.maxExpansion = 10000
        self.time_begin = 0

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

    def maxValue(
        self, state: GameState, alpha, beta, theta=0, depth=0
    ) -> (float, Action, int):
        # If one find a terminal state, the search is done
        if state.is_done():
            return (0, None, theta)

        # Initialize optimal score
        v_star = -np.inf
        v = None
        # Initialize optimal move
        m_star = None
        # Get the best 5 actions
        possible_actions = self.actions_to_expore(state)

        for action in possible_actions:
            # Get Next state
            next_state = action.get_next_game_state()

            # # Max time reached
            # if time.time() - self.time_begin > 15:
            #     v = self.heuristique(next_state)
            if theta >= self.maxExpansion:
                v = self.heuristique(next_state)
            # Expansion of node's branches
            else:
                # With transposition tables
                if TRANSPO:
                    next_state_hash = self.zobrist_hash(next_state)
                    if not next_state_hash in self.visited_positions:
                        self.visited_positions.append(next_state_hash)
                        theta += 1
                        depth += 1
                        v, _, theta = self.minValue(
                            next_state, alpha, beta, theta, depth
                        )
                # Without Transpo tables
                else:
                    theta += 1
                    depth += 1
                    v, _, theta = self.minValue(next_state, alpha, beta, theta)

            if v and v > v_star:
                v_star = v
                m_star = action
                alpha = max(alpha, v_star)

            # Pruning
            if v_star >= beta:
                return (v_star, m_star, theta)

        return (v_star, m_star, theta)

    def minValue(
        self, state: GameState, alpha, beta, theta=0, depth=0
    ) -> (float, Action, int):
        # Terminal State
        if state.is_done():
            return (-6, None, theta)

        # Initialization of optimal score
        v_star = np.inf
        v = None
        # Initialization of optimal move
        m_star = None
        # Get best 5 actions
        possible_actions = self.actions_to_expore(state)
        for action in possible_actions:
            # Get next state
            next_state = action.get_next_game_state()

            # Max time reached
            # if time.time() - self.time_begin > 15:
            #     v = self.heuristique(next_state)
            if theta >= self.maxExpansion:
                v = self.heuristique(next_state)

            # Expansion of node's children
            else:
                if TRANSPO:
                    next_state_hash = self.zobrist_hash(next_state)
                    if not next_state_hash in self.visited_positions:
                        self.visited_positions.append(next_state_hash)
                        theta += 1
                        depth += 1
                        v, _, theta = self.maxValue(
                            next_state, alpha, beta, theta, depth
                        )
                else:
                    theta += 1
                    depth += 1
                    v, _, theta = self.maxValue(next_state, alpha, beta, theta)

            if v and v < v_star:
                v_star = v
                m_star = action
                beta = min(beta, v_star)

            # Pruning
            if v_star <= alpha:
                return (v_star, m_star, theta)
        return (v_star, m_star, theta)

    def actions_to_expore(self, state: GameState, nbActions: int = 5) -> list[Action]:
        possible_actions = list(state.get_possible_actions())
        actions_heuristics = [
            self.heuristique(action.get_next_game_state())
            for action in possible_actions
        ]
        best_actions_indexes = [np.argsort(actions_heuristics)[-i] for i in range(1, 6)]
        return [possible_actions[i] for i in best_actions_indexes]

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        self.time_begin = time.time()
        if self.first_action:
            self.init_zobrist(current_state.rep)

        _, action, _ = self.maxValue(current_state, -np.inf, np.inf)
        self.first_action = False

        print("time elapsed for decision process: ", time.time() - self.time_begin)
        self.visited_positions = []
        return action


TRANSPO = True
