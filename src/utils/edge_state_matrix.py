from __future__ import annotations
import numpy as np
import networkx as nx


class EdgeState:
    """
    Different states that a directed edge can be in.
    """

    BANNED = -1  # Not in the graph and defintiely should not be.
    ABSENT = 0  # Not in the graph but could go either way.
    PRESENT = 1  # In the graph by the user's suggestion, but could go either way.
    SUGGESTED = 2  # In the graph by the system's suggestion, but could go either way.
    FIXED = 3  # In the graph and definitely should be.


class EdgeStateMatrix:
    """
    A class for managing an edge state matrix.

    An edge state matrix is square, with the entry (i,j) representing the state
    of the directed edge between nodes i and j.

    Self-edges are not allowed. Fixing an edge bans its inverse.
    """

    def __init__(
        self, variables: list[str], default_state: EdgeState = EdgeState.ABSENT
    ) -> None:
        """
        Initialize the edge state matrix to the right dimensions and ban all self-edges.
        The rest of the edges are set to the provided default state.

        Parameters:
            variables: The variables to initialize the edge state matrix based on.
            default_state: The default state to set the edges to.
        """

        n = len(variables)
        self._variables = variables
        self._m = np.full((n, n), default_state)

        for i in range(n):
            self._m[i, i] = EdgeState.BANNED

    @property
    def m(self) -> np.ndarray:
        """
        Returns the edge state matrix.
        """
        return self._m

    @property
    def n(self) -> int:
        """
        Returns the number of nodes.
        """
        return self._m.shape[0]

    def clear_and_set_from_graph(
        self,
        graph: nx.DiGraph,
        state_for_edges_in_graph: EdgeState = EdgeState.PRESENT,
        state_for_edges_not_in_graph: EdgeState = EdgeState.ABSENT,
    ) -> None:
        """
        Clear the edge state matrix and then set it based on the provided graph
        and edge states. Inverses of edges in the graph, as well as self-edges,
        are banned.

        Parameters:
            graph: The graph to use to set the edge states.
            state_for_edges_in_graph: The state to use for edges that are in the graph.
            state_for_edges_not_in_graph: The state to use for edges that are not in the graph.
        """

        self._m = np.full((self.n, self.n), state_for_edges_not_in_graph)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self._m[i, j] = EdgeState.BANNED
                elif graph.has_edge(self.name(i), self.name(j)):
                    self._m[i, j] = state_for_edges_in_graph

    def clear_and_set_from_matrix(self, m: np.ndarray) -> None:
        """
        Clear the edge state matrix and then set it based on the provided matrix.

        Parameters:
            m: The matrix to use to set the edge states.
        """

        self._m = m

    def idx(self, var: str) -> int:
        """
        Retrieve the index of a variable in the edge state matrix.

        Parameters:
            var: The name of the variable.

        Returns:
            The index of the variable in the edge state matrix.
        """
        return self._variables.index(var)

    def name(self, idx: int) -> str:
        """
        Retrieve the name of a variable in the edge state matrix.

        Parameters:
            idx: The index of the variable.

        Returns:
            The name of the variable in the edge state matrix.
        """
        return self._variables[idx]

    def get_edge_state(self, src: str | int, dst: str | int) -> EdgeState:
        """
        Get the state of a specific edge.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.

        Returns:
            The state of the edge.
        """
        src_idx = self.idx(src) if type(src) == str else src
        dst_idx = self.idx(dst) if type(dst) == str else dst
        return self._m[src_idx][dst_idx]

    def is_edge_in_state(
        self, src: str | int, dst: str | int, state: EdgeState
    ) -> bool:
        """
        Check if an edge is in a specific state.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.
            state: The state to check for.

        Returns:
            True if the edge is in the specified state, False otherwise.
        """
        src_idx = self.idx(src) if type(src) == str else src
        dst_idx = self.idx(dst) if type(dst) == str else dst
        return self.get_edge_state(src, dst) == state

    def is_edge_fixed(self, src: str | int, dst: str | int) -> bool:
        """
        Check if an edge is fixed.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.

        Returns:
            True if the edge is fixed, False otherwise.
        """
        return self.is_edge_in_state(src, dst, EdgeState.FIXED)

    def is_edge_banned(self, src: str | int, dst: str | int) -> bool:
        """
        Check if an edge is banned.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.

        Returns:
            True if the edge is banned, False otherwise.
        """
        return self.is_edge_in_state(src, dst, EdgeState.BANNED)

    def mark_edge(self, src: str | int, dst: str | int, state: EdgeState) -> None:
        """
        Mark an edge as being in a specified state. Fixing an edge bans its inverse.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.
            state: The state to mark the edge with.
        """

        src_idx = self.idx(src) if type(src) == str else src
        dst_idx = self.idx(dst) if type(dst) == str else dst

        self._m[src_idx][dst_idx] = state
        if state == EdgeState.FIXED:
            self._m[dst_idx][src_idx] = EdgeState.BANNED

    def _all_edges_in_state(self, state: EdgeState) -> list[tuple[int, int]]:
        """
        Get a list of all edges in a specific state.

        Parameters:
            state: The state to check for.

        Returns:
            A list of all edges in the specified state.
        """
        rows, cols = np.asarray(self.m == state).nonzero()
        return [(self.name(rows[i]), self.name(cols[i])) for i in range(len(rows))]

    @property
    def fixed_list(self) -> list[tuple[str, str]]:
        """
        Get a list of all edges that are fixed.

        Returns:
            A list of all edges that are fixed.
        """
        return self._all_edges_in_state(EdgeState.FIXED)

    @property
    def ban_list(self) -> list[tuple[str, str]]:
        """
        Get a list of all edges that are banned.

        Returns:
            A list of all edges that are banned.
        """
        return self._all_edges_in_state(EdgeState.BANNED)

    # Implement a copy method
    def copy(self):
        """
        Create a copy of the edge state matrix.

        Returns:
            A copy of the edge state matrix.
        """
        new_matrix = EdgeStateMatrix(self._variables)
        new_matrix.clear_and_set_from_matrix(self._m)
        return new_matrix
