"""
ReshareClassifier — the 4x3 Breadth x Depth structural grid (RQ1 plan,
Section 2 "Lens 1: Structural Fidelity Evaluation").

Breadth categories (4), based on total reshares N_r:
    Single        : N_r == 1
    Non-Extensive : 2 <= N_r <= 5
    Extensive     : N_r > 5  AND depth <= 2   (shallow-but-wide burst)
    Cascading     : N_r > 5  AND depth >= 3   (wide AND deep)

Depth categories (3), based on cascade depth D_c:
    Shallow  : D_c <= 2
    Moderate : 3 <= D_c <= 4
    Deep     : D_c > 4

*** Design note worth flagging in your methodology chapter ***
Two of the four breadth categories (Extensive, Cascading) are already
conditioned on depth in their own definition, which means the 4x3 grid
is NOT a free cross-product -- 3 of the 12 cells are structurally
IMPOSSIBLE and will always read zero, regardless of how good the graph
construction is:
    Extensive  x Moderate  -- impossible (Extensive requires depth<=2)
    Extensive  x Deep      -- impossible (Extensive requires depth<=2)
    Cascading  x Shallow   -- impossible (Cascading requires depth>=3)
This isn't a bug in this implementation -- it falls directly out of how
the plan defines Extensive/Cascading -- but you should show the grid
with those 3 cells explicitly marked "N/A" rather than "0", or a
reviewer may read them as a graph-construction failure rather than a
structural impossibility. `grid_to_table()` below marks them for you.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import torch


class Breadth(Enum):
    SINGLE = "Single"
    NON_EXTENSIVE = "Non-Extensive"
    EXTENSIVE = "Extensive"
    CASCADING = "Cascading"


class Depth(Enum):
    SHALLOW = "Shallow"
    MODERATE = "Moderate"
    DEEP = "Deep"


IMPOSSIBLE_CELLS = {
    (Breadth.EXTENSIVE, Depth.MODERATE),
    (Breadth.EXTENSIVE, Depth.DEEP),
    (Breadth.CASCADING, Depth.SHALLOW),
}


@dataclass
class CascadeStats:
    n_reshares: int   # N_r -- total reshare count
    depth: int        # D_c -- max propagation depth


def classify_depth(depth: int) -> Depth:
    if depth <= 2:
        return Depth.SHALLOW
    if depth <= 4:
        return Depth.MODERATE
    return Depth.DEEP


def classify_breadth(n_reshares: int, depth: int) -> Breadth:
    # NOTE: the plan defines Single as "N_r == 1" and doesn't say what
    # happens at N_r == 0 (a cascade with organic replies but zero
    # resharing at all -- a real, common case, not an edge case).
    # Treating N_r == 0 as Non-Extensive would be wrong (that category
    # is explicitly "2 <= N_r <= 5"). We fold N_r == 0 into Single
    # instead (both are "essentially no resharing activity"), which
    # keeps the grid at 4 breadth categories as specified. Flag this
    # interpretation in your methodology chapter -- it's a genuine gap
    # in the plan as written, not an arbitrary implementation choice.
    if n_reshares <= 1:
        return Breadth.SINGLE
    if n_reshares <= 5:
        return Breadth.NON_EXTENSIVE
    # n_reshares > 5 from here
    if depth <= 2:
        return Breadth.EXTENSIVE
    return Breadth.CASCADING  # depth >= 3 implied by not being Extensive


class ReshareClassifier:
    """Bins a batch of cascades into the 12 (Breadth x Depth) cells."""

    @staticmethod
    def classify(stats: CascadeStats) -> Tuple[Breadth, Depth]:
        return (classify_breadth(stats.n_reshares, stats.depth),
                classify_depth(stats.depth))

    @staticmethod
    def classify_batch(stats_list: List[CascadeStats]) -> List[Tuple[Breadth, Depth]]:
        return [ReshareClassifier.classify(s) for s in stats_list]

    @staticmethod
    def stats_from_duplication_graph(graph) -> CascadeStats:
        """Derive N_r and D_c directly from a DuplicationGraph (post
        graph-construction) -- used to check whether the CONSTRUCTED
        graph still reflects the true cascade shape."""
        n_reshares = int(graph.duplication_freq.sum().item())
        depth = int(graph.propagation_depth.max().item()) if graph.propagation_depth.numel() else 0
        return CascadeStats(n_reshares=n_reshares, depth=depth)

    @staticmethod
    def stats_from_raw_posts(posts) -> CascadeStats:
        """Ground-truth N_r / D_c computed directly from the raw
        RawPost list, independent of any graph-construction choice --
        this is the reference each construction strategy is compared
        against in Lens 1."""
        from .duplication_graph import _is_duplicate
        n_reshares = 0
        seen = set()
        for p in posts:
            if p.post_id in seen:
                continue
            group = [q for q in posts if q.post_id not in seen
                     and _is_duplicate(p.text, q.text)]
            seen.update(q.post_id for q in group)
            n_reshares += max(0, len(group) - 1)

        # Depth via BFS DOWN from the root over children edges (same
        # pattern as data/duplication_graph.py's own depth computation)
        # rather than recursing UP parent pointers. Recursing up is
        # fragile in two ways real data actually hits: (1) it blows
        # Python's recursion limit on long reply chains, and (2) if
        # any cascade has a cycle in its parent links (two comments
        # pointing to each other, or a chain that loops back on
        # itself -- a real data-quality issue, not just theoretical),
        # it recurses forever instead of terminating. BFS from the
        # root with a visited-set can't do either: cycles are simply
        # ignored (a node already visited is never re-queued), and
        # there's no recursion depth to exceed.
        root = next((p for p in posts if p.parent_id is None), None)
        if root is None:
            # No explicit root in this cascade -- can't compute a
            # meaningful depth. Return depth 0 rather than crashing;
            # this cascade's data is malformed upstream of this code.
            return CascadeStats(n_reshares=n_reshares, depth=0)

        children = {p.post_id: [] for p in posts}
        for p in posts:
            if p.parent_id is not None and p.parent_id in children:
                children[p.parent_id].append(p.post_id)

        depth = {root.post_id: 0}
        visited = {root.post_id}
        frontier = [root.post_id]
        while frontier:
            nxt = []
            for pid in frontier:
                for c in children.get(pid, []):
                    if c in visited:
                        continue  # cycle guard: never re-visit a node
                    visited.add(c)
                    depth[c] = depth[pid] + 1
                    nxt.append(c)
            frontier = nxt

        # Nodes never reached from the root (e.g. isolated inside a
        # cycle that doesn't touch the root) are silently excluded --
        # they're a data-quality issue in the source file, not
        # something this function should crash over.
        max_depth = max(depth.values()) if depth else 0
        return CascadeStats(n_reshares=n_reshares, depth=max_depth)


def build_grid(labels: List[Tuple[Breadth, Depth]]) -> dict:
    """Counts cascades per (Breadth, Depth) cell."""
    grid = {(b, d): 0 for b in Breadth for d in Depth}
    for b, d in labels:
        grid[(b, d)] += 1
    return grid


def grid_to_table(grid: dict) -> str:
    breadths = list(Breadth)
    depths = list(Depth)
    header = f"{'':<16}" + "".join(f"{d.value:<12}" for d in depths)
    lines = [header]
    for b in breadths:
        row = f"{b.value:<16}"
        for d in depths:
            if (b, d) in IMPOSSIBLE_CELLS:
                row += f"{'N/A':<12}"
            else:
                row += f"{grid.get((b, d), 0):<12}"
        lines.append(row)
    return "\n".join(lines)
