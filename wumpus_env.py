# wumpus_env.py
# =========================================================
# Wumpus World Environment (Russell & Norvig style)
# ---------------------------------------------------------
# Purpose:
#   This file defines the *environment* (the "world") and the
#   data structures used by agents (actions + percepts).
#
# Key design principle:
#   - The agent should ONLY receive Percepts (stench, breeze,
#     glitter, bump, scream, reward).
#   - The full board (pits/wumpus/gold locations) is hidden and
#     only exposed via get_debug_board() for visualization.
#
# Compatible with:
#   - Streamlit UI (animation)
#   - Jupyter visualization
#   - Batch simulations
# =========================================================

from __future__ import annotations
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Set, Dict, List, Tuple


# =========================================================
# Enums: Actions and Directions
# =========================================================

class Action(Enum):
    """
    The set of legal actions in the Wumpus World.
    These match the standard actions from the textbook.
    """
    FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    SHOOT = auto()
    GRAB = auto()
    CLIMB = auto()


class Direction(Enum):
    """
    Agent orientation in the grid.
    The environment uses direction to interpret FORWARD.
    """
    N = auto()
    E = auto()
    S = auto()
    W = auto()

    def left(self) -> "Direction":
        """Return the direction after a 90-degree left turn."""
        return {
            Direction.N: Direction.W,
            Direction.W: Direction.S,
            Direction.S: Direction.E,
            Direction.E: Direction.N,
        }[self]

    def right(self) -> "Direction":
        """Return the direction after a 90-degree right turn."""
        return {
            Direction.N: Direction.E,
            Direction.E: Direction.S,
            Direction.S: Direction.W,
            Direction.W: Direction.N,
        }[self]


# =========================================================
# Data Structures: Position and Percept
# =========================================================

@dataclass(frozen=True)
class Pos:
    """
    2D grid position (x, y).
    Coordinates start at (1,1) in the bottom-left corner.
    """
    x: int
    y: int

    def neighbors_4(self) -> List["Pos"]:
        """
        Return the 4-neighborhood (Von Neumann neighborhood):
        up, down, left, right.
        The caller must still check bounds.
        """
        return [
            Pos(self.x + 1, self.y),
            Pos(self.x - 1, self.y),
            Pos(self.x, self.y + 1),
            Pos(self.x, self.y - 1),
        ]


@dataclass
class Percept:
    """
    The Percept is the ONLY information an agent is supposed to observe.
    This is the classical Wumpus World percept vector:
      - stench  : True if Wumpus is in adjacent cell (or same cell)
      - breeze  : True if a pit is adjacent
      - glitter : True if gold is in current cell
      - bump    : True if last FORWARD hit a wall
      - scream  : True if Wumpus was killed by SHOOT
      - reward  : numeric reward for the last action
    """
    stench: bool
    breeze: bool
    glitter: bool
    bump: bool
    scream: bool
    reward: int

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool, int]:
        """Convenience method for debugging/printing."""
        return (
            self.stench,
            self.breeze,
            self.glitter,
            self.bump,
            self.scream,
            self.reward,
        )


# =========================================================
# Environment: WumpusWorldEnv
# =========================================================

class WumpusWorldEnv:
    """
    Environment simulator for Wumpus World.

    Separation of responsibilities:
      - The environment stores the complete world state and applies rules.
      - The agent chooses actions using ONLY percepts.
      - Debug board is available only for visualization/UI.

    Reward structure (typical):
      - -1 per action (living cost / encourages shorter plans)
      - -1000 if the agent dies (pit or live Wumpus)
      - -10 for shooting an arrow
      - +1000 for climbing out with gold (success)
    """

    def __init__(
        self,
        width: int = 4,
        height: int = 4,
        allow_climb_without_gold: bool = True,
        pit_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new random world.

        Parameters
        ----------
        width, height:
            Grid dimensions (default 4x4).
        allow_climb_without_gold:
            If True, the agent may CLIMB from the start even without gold.
            (Some variants require having gold to successfully exit.)
        pit_prob:
            Probability that each non-start cell contains a pit.
        seed:
            RNG seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.allow_climb_without_gold = allow_climb_without_gold
        self.pit_prob = pit_prob
        self.rng = random.Random(seed)

        # -------------------------
        # Agent internal state
        # -------------------------
        self.agent_pos = Pos(1, 1)       # start at bottom-left
        self.agent_dir = Direction.E     # default facing East

        # -------------------------
        # World objects
        # -------------------------
        # Note: gold and wumpus are placed independently and could overlap
        # (this is a design choice; some versions disallow overlaps).
        self.gold_pos = self._random_non_start_pos()
        self.wumpus_pos = self._random_non_start_pos()
        self.wumpus_alive = True

        # Pits are a set of positions
        self.pits: Set[Pos] = set()
        self._place_pits()

        # -------------------------
        # Inventory state
        # -------------------------
        self.has_gold = False
        self.arrow_available = True

        # -------------------------
        # Episode terminal state
        # -------------------------
        self.terminated = False
        self.dead = False

        # Transient signals for percepts
        # (bump only after hitting wall, scream only after killing Wumpus)
        self._last_bump = False
        self._last_scream = False

        # Metrics
        self.steps = 0
        self.total_reward = 0

    # =====================================================
    # Internal helpers (not exposed to agent)
    # =====================================================

    def _in_bounds(self, p: Pos) -> bool:
        """Check whether a position is inside the grid."""
        return 1 <= p.x <= self.width and 1 <= p.y <= self.height

    def _random_non_start_pos(self) -> Pos:
        """
        Sample a random position excluding the start cell (1,1).
        Used for Wumpus and gold placement.
        """
        cells = [
            Pos(x, y)
            for x in range(1, self.width + 1)
            for y in range(1, self.height + 1)
            if not (x == 1 and y == 1)
        ]
        return self.rng.choice(cells)

    def _place_pits(self):
        """
        Place pits independently in each non-start cell with probability pit_prob.
        This matches the typical randomized Wumpus World generation.
        """
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                p = Pos(x, y)
                if p != Pos(1, 1) and self.rng.random() < self.pit_prob:
                    self.pits.add(p)

    def _forward_pos(self, p: Pos, d: Direction) -> Pos:
        """
        Compute the next position if the agent moves FORWARD from position p
        while facing direction d.
        """
        if d == Direction.N:
            return Pos(p.x, p.y + 1)
        if d == Direction.S:
            return Pos(p.x, p.y - 1)
        if d == Direction.E:
            return Pos(p.x + 1, p.y)
        return Pos(p.x - 1, p.y)

    # =====================================================
    # Public API (used by UI / simulations)
    # =====================================================

    def is_terminal(self) -> bool:
        """Return True if the episode has ended."""
        return self.terminated

    def get_percept(self, reward_override: int = 0) -> Percept:
        """
        Construct the current percept based on the agent's position and
        the transient signals from the previous action.

        Important:
          - This is what an agent is supposed to observe.
          - reward_override is set by step() to report the last action reward.
        """
        return Percept(
            stench=self._stench_at(self.agent_pos),
            breeze=self._breeze_at(self.agent_pos),
            glitter=(self.agent_pos == self.gold_pos),
            bump=self._last_bump,
            scream=self._last_scream,
            reward=reward_override,
        )

    def step(self, action: Action) -> Percept:
        """
        Apply one action and return the resulting percept (including reward).

        Rules:
          - Each action costs -1.
          - SHOOT costs -10 extra and can kill the Wumpus if it is in line.
          - GRAB picks up gold if in the same cell.
          - CLIMB ends episode if at start (optionally requires gold).
          - If agent enters a pit or a live Wumpus cell, agent dies (-1000).
        """
        if self.terminated:
            # Once terminal, additional steps do nothing
            return self.get_percept(0)

        # Update counters and reset transient percept flags
        self.steps += 1
        self._last_bump = False
        self._last_scream = False

        # Default action cost
        reward = -1

        # -------------------------
        # Execute action
        # -------------------------
        if action == Action.TURN_LEFT:
            self.agent_dir = self.agent_dir.left()

        elif action == Action.TURN_RIGHT:
            self.agent_dir = self.agent_dir.right()

        elif action == Action.FORWARD:
            nxt = self._forward_pos(self.agent_pos, self.agent_dir)
            if self._in_bounds(nxt):
                self.agent_pos = nxt
            else:
                # Wall collision: agent stays in place but receives bump percept
                self._last_bump = True

        elif action == Action.GRAB:
            # Gold is collected only if agent is on the gold cell
            if self.agent_pos == self.gold_pos:
                self.has_gold = True

        elif action == Action.SHOOT:
            # Arrow can be shot once
            if self.arrow_available:
                self.arrow_available = False
                reward -= 10  # extra cost for using arrow
                if self._shoot_arrow():
                    # If Wumpus is in line of fire, it dies and we emit a scream percept
                    self.wumpus_alive = False
                    self._last_scream = True

        elif action == Action.CLIMB:
            # Climb is only possible from the starting cell
            if self.agent_pos == Pos(1, 1):
                if self.allow_climb_without_gold or self.has_gold:
                    # Success reward only if agent actually has gold
                    if self.has_gold:
                        reward += 1000
                    self.terminated = True

        # -------------------------
        # Death check (if episode not already ended by CLIMB)
        # -------------------------
        if not self.terminated:
            if self.agent_pos in self.pits:
                reward -= 1000
                self.dead = True
                self.terminated = True
            elif self.agent_pos == self.wumpus_pos and self.wumpus_alive:
                reward -= 1000
                self.dead = True
                self.terminated = True

        # Update cumulative score and return percept
        self.total_reward += reward
        return self.get_percept(reward)

    # =====================================================
    # Mechanics: Shooting
    # =====================================================

    def _shoot_arrow(self) -> bool:
        """
        Simulate arrow flight in a straight line in the agent's facing direction.
        Returns True if the arrow hits the Wumpus while it is alive.
        """
        p = self.agent_pos
        while True:
            p = self._forward_pos(p, self.agent_dir)
            if not self._in_bounds(p):
                return False
            if p == self.wumpus_pos and self.wumpus_alive:
                return True

    # =====================================================
    # Percept generation functions
    # =====================================================

    def _stench_at(self, p: Pos) -> bool:
        """
        Stench is True if the Wumpus is in an adjacent cell
        (or in the same cell, depending on implementation).
        """
        if p == self.wumpus_pos:
            return True
        return any(n == self.wumpus_pos for n in p.neighbors_4() if self._in_bounds(n))

    def _breeze_at(self, p: Pos) -> bool:
        """Breeze is True if any adjacent cell contains a pit."""
        return any(n in self.pits for n in p.neighbors_4() if self._in_bounds(n))

    # =====================================================
    # Debug board (Visualization only)
    # =====================================================

    def get_debug_board(self) -> Dict[Pos, Set[str]]:
        """
        Return a full map of the environment for UI/debugging.

        IMPORTANT:
          - This should NOT be used by a "proper" agent, because it exposes
            hidden information (pits, exact Wumpus location, gold location).
          - It is safe for visualization (Streamlit board).
        """
        board: Dict[Pos, Set[str]] = {
            Pos(x, y): set()
            for x in range(1, self.width + 1)
            for y in range(1, self.height + 1)
        }
        for pit in self.pits:
            board[pit].add("P")
        board[self.gold_pos].add("G")
        board[self.wumpus_pos].add("W" if self.wumpus_alive else "w")
        board[self.agent_pos].add("A")
        return board


# =========================================================
# Naive Agent (Baseline)
# =========================================================

class NaiveAgent:
    """
    A baseline agent that chooses actions uniformly at random.

    Educational purpose:
      - Demonstrates how an agent interacts with the environment
        through percepts only.
      - Provides a baseline for comparing smarter agents later.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.actions = list(Action)

    def choose_action(self, percept: Percept) -> Action:
        """
        Choose an action using ONLY the percept (no access to the board).
        This agent ignores the percept and acts randomly.
        """
        return self.rng.choice(self.actions)
