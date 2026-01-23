"""
holos/games/chess_targeted.py - Targeted Chess Search with Material Filtering

This module extends ChessGame to support targeted searches with specific goals:
- Search only for paths leading to a specific material configuration
- Early terminate paths that capture to wrong material
- Generate 8-piece positions that could reach target 7-piece material

Example use case: Find ALL 8-piece positions leading to KQRRvKQR 7-piece solutions,
excluding paths that capture to other material configurations.

This demonstrates HOLOS's ability to define sub-goals and intermediate targets
beyond just terminal checkmates.
"""

import random
from typing import List, Tuple, Optional, Any, Set, Dict, FrozenSet
from dataclasses import dataclass
from collections import defaultdict

from holos.games.chess import (
    ChessGame, ChessState, ChessValue, Piece, PIECE_VALUES,
    generate_moves, apply_move, generate_predecessors,
    in_check, piece_type, is_terminal, SyzygyProbe,
    extract_features, PIECE_NAMES
)


def get_material_signature(state: ChessState) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get material signature as (white_pieces, black_pieces) sorted tuples.

    Example: KQRRvKQR -> ((1, 2, 4, 4), (7, 8, 10))  # King=1/7, Queen=2/8, Rook=4/10
    """
    white = tuple(sorted(p for p, sq in state.pieces if p <= 6))
    black = tuple(sorted(p for p, sq in state.pieces if p > 6))
    return (white, black)


def material_string(state: ChessState) -> str:
    """Get human-readable material string like 'KQRRvKQR'"""
    piece_chars = {1: 'K', 2: 'Q', 3: 'R', 4: 'R', 5: 'B', 6: 'N',
                   7: 'K', 8: 'Q', 9: 'R', 10: 'R', 11: 'B', 12: 'N'}
    # Map piece values to characters
    reverse_map = {
        Piece.W_KING: 'K', Piece.W_QUEEN: 'Q', Piece.W_ROOK: 'R',
        Piece.W_BISHOP: 'B', Piece.W_KNIGHT: 'N', Piece.W_PAWN: 'P',
        Piece.B_KING: 'k', Piece.B_QUEEN: 'q', Piece.B_ROOK: 'r',
        Piece.B_BISHOP: 'b', Piece.B_KNIGHT: 'n', Piece.B_PAWN: 'p',
    }
    white = ''.join(sorted([reverse_map.get(p, '?') for p, sq in state.pieces if p <= 6],
                           key=lambda c: 'KQRBNP'.index(c) if c in 'KQRBNP' else 99))
    black = ''.join(sorted([reverse_map.get(p, '?') for p, sq in state.pieces if p > 6],
                           key=lambda c: 'kqrbnp'.index(c) if c in 'kqrbnp' else 99))
    return f"{white}v{black}".upper()


def parse_material_string(material: str) -> Tuple[List[int], List[int]]:
    """Parse 'KQRRvKQR' into piece lists"""
    material = material.upper()
    white_str, black_str = material.split('V')

    white_pieces = []
    for c in white_str:
        if c == 'K': white_pieces.append(Piece.W_KING)
        elif c == 'Q': white_pieces.append(Piece.W_QUEEN)
        elif c == 'R': white_pieces.append(Piece.W_ROOK)
        elif c == 'B': white_pieces.append(Piece.W_BISHOP)
        elif c == 'N': white_pieces.append(Piece.W_KNIGHT)
        elif c == 'P': white_pieces.append(Piece.W_PAWN)

    black_pieces = []
    for c in black_str:
        if c == 'K': black_pieces.append(Piece.B_KING)
        elif c == 'Q': black_pieces.append(Piece.B_QUEEN)
        elif c == 'R': black_pieces.append(Piece.B_ROOK)
        elif c == 'B': black_pieces.append(Piece.B_BISHOP)
        elif c == 'N': black_pieces.append(Piece.B_KNIGHT)
        elif c == 'P': black_pieces.append(Piece.B_PAWN)

    return white_pieces, black_pieces


def get_8piece_variants(target_7piece: str) -> List[str]:
    """
    Get all 8-piece material configs that could capture down to target 7-piece.

    Example: 'KQRRvKQR' can come from:
    - KQRRvKQRR (white captures black rook)
    - KQRRvKQRQ (white captures black queen)
    - KQRRvKQRB (white captures black bishop)
    - etc for all piece types
    - KQRRRvKQR (black captures white rook)
    - etc
    """
    white_str, black_str = target_7piece.upper().split('V')
    variants = []

    # Add piece to black (white will capture it)
    for piece in ['Q', 'R', 'B', 'N', 'P']:
        variants.append(f"{white_str}v{black_str}{piece}")

    # Add piece to white (black will capture it)
    for piece in ['Q', 'R', 'B', 'N', 'P']:
        variants.append(f"{white_str}{piece}v{black_str}")

    return variants


class TargetedChessGame(ChessGame):
    """
    Chess game with targeted material filtering.

    Only considers positions leading to a specific material configuration
    as "boundary" positions. Captures to other materials are filtered out.
    """

    def __init__(self, syzygy_path: str = "./syzygy",
                 target_material: str = "KQRRvKQR",
                 source_materials: List[str] = None):
        """
        Args:
            syzygy_path: Path to syzygy tablebases
            target_material: The 7-piece material we're targeting (e.g., "KQRRvKQR")
            source_materials: 8-piece materials to search from (auto-generated if None)
        """
        # Target is 7-piece, source is 8-piece
        target_pieces = len(target_material.replace('V', '').replace('v', ''))
        super().__init__(syzygy_path, min_pieces=target_pieces, max_pieces=target_pieces + 1)

        self.target_material = target_material.upper()
        self.target_signature = self._parse_signature(self.target_material)

        # Generate valid 8-piece source materials if not provided
        if source_materials is None:
            source_materials = get_8piece_variants(target_material)
        self.source_materials = [m.upper() for m in source_materials]
        self.source_signatures = {self._parse_signature(m) for m in self.source_materials}

        # Stats for filtering
        self.filter_stats = {
            'wrong_material_filtered': 0,
            'target_material_found': 0,
            'source_positions_generated': 0,
        }

        print(f"TargetedChessGame initialized:")
        print(f"  Target: {self.target_material} ({self.min_pieces} pieces)")
        print(f"  Sources: {len(self.source_materials)} 8-piece configurations")

    def _parse_signature(self, material: str) -> FrozenSet[Tuple[str, int]]:
        """Parse material string into a hashable signature"""
        white_str, black_str = material.upper().split('V')
        pieces = []
        for c in white_str:
            pieces.append(('w', c))
        for c in black_str:
            pieces.append(('b', c))
        return frozenset(pieces)

    def _get_signature(self, state: ChessState) -> FrozenSet[Tuple[str, int]]:
        """Get signature from state"""
        pieces = []
        piece_chars = {
            Piece.W_KING: 'K', Piece.W_QUEEN: 'Q', Piece.W_ROOK: 'R',
            Piece.W_BISHOP: 'B', Piece.W_KNIGHT: 'N', Piece.W_PAWN: 'P',
            Piece.B_KING: 'K', Piece.B_QUEEN: 'Q', Piece.B_ROOK: 'R',
            Piece.B_BISHOP: 'B', Piece.B_KNIGHT: 'N', Piece.B_PAWN: 'P',
        }
        for p, sq in state.pieces:
            color = 'w' if p <= 6 else 'b'
            char = piece_chars.get(p, '?')
            pieces.append((color, char))
        return frozenset(pieces)

    def is_target_material(self, state: ChessState) -> bool:
        """Check if state has target material configuration"""
        return self._get_signature(state) == self.target_signature

    def is_source_material(self, state: ChessState) -> bool:
        """Check if state has a valid source material configuration"""
        return self._get_signature(state) in self.source_signatures

    def is_boundary(self, state: ChessState) -> bool:
        """
        Boundary = target material AND syzygy can solve it.

        This is more restrictive than base ChessGame - we only accept
        positions with the exact target material.
        """
        if state.piece_count() != self.min_pieces:
            return False
        if not self.is_target_material(state):
            return False
        return True

    def get_boundary_value(self, state: ChessState) -> Optional[ChessValue]:
        """Get value only if it's target material"""
        if not self.is_boundary(state):
            return None
        val = self.syzygy.probe(state)
        if val is not None:
            self.filter_stats['target_material_found'] += 1
            return ChessValue(val)
        return None

    def get_successors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """
        Get successors, filtering out captures to wrong material.

        If a move is a capture:
        - If result is target material: allow (this is what we want!)
        - If result is wrong 7-piece material: filter out (dead end)
        - If result is still 8-piece: allow (no capture yet)
        """
        if state.piece_count() > self.max_pieces:
            return []

        moves = generate_moves(state)
        successors = []

        for move in moves:
            child = apply_move(state, move)
            child_count = child.piece_count()

            # Non-capture move
            if move[2] is None:
                if child_count <= self.max_pieces:
                    successors.append((child, move))
                continue

            # Capture move - check resulting material
            if child_count == self.min_pieces:
                # Captured down to 7 pieces - must be target material
                if self.is_target_material(child):
                    successors.append((child, move))
                else:
                    self.filter_stats['wrong_material_filtered'] += 1
            elif child_count > self.min_pieces:
                # Still above target - allow
                successors.append((child, move))

        return successors

    def get_predecessors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """
        Get predecessors (unmoves), only allowing those from valid source materials.
        """
        if state.piece_count() >= self.max_pieces:
            return []

        preds = generate_predecessors(state, max_uncaptures=3)
        valid_preds = []

        for pred in preds:
            if pred.piece_count() == self.max_pieces:
                # This is an 8-piece position - must be valid source material
                if self.is_source_material(pred):
                    valid_preds.append((pred, None))
                else:
                    self.filter_stats['wrong_material_filtered'] += 1
            elif pred.piece_count() < self.max_pieces:
                valid_preds.append((pred, None))

        return valid_preds

    def generate_target_boundary_seeds(self, count: int = 1000) -> List[ChessState]:
        """
        Generate random positions with target material for backward seeding.

        These are the 7-piece positions we're trying to reach.
        """
        white_pieces, black_pieces = parse_material_string(self.target_material)
        all_pieces = white_pieces + black_pieces
        positions = []

        for _ in range(count * 10):
            if len(positions) >= count:
                break

            squares = random.sample(range(64), len(all_pieces))
            pieces = list(zip(all_pieces, squares))
            state = ChessState(pieces, random.choice(['w', 'b']))

            # Validate
            wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
            bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
            if wk is None or bk is None:
                continue
            if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
                continue
            if in_check(state, 'b' if state.turn == 'w' else 'w'):
                continue

            # Must be solvable by syzygy
            if self.syzygy.probe(state) is not None:
                positions.append(state)

        print(f"Generated {len(positions)} target boundary positions ({self.target_material})")
        return positions

    def generate_source_positions(self, count_per_material: int = 100) -> List[ChessState]:
        """
        Generate 8-piece positions from all valid source materials.

        These are the starting positions for forward search.
        """
        all_positions = []

        for material in self.source_materials:
            white_pieces, black_pieces = parse_material_string(material)
            all_pieces = white_pieces + black_pieces
            positions = []

            for _ in range(count_per_material * 10):
                if len(positions) >= count_per_material:
                    break

                squares = random.sample(range(64), len(all_pieces))
                pieces = list(zip(all_pieces, squares))
                state = ChessState(pieces, random.choice(['w', 'b']))

                # Validate
                wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
                bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
                if wk is None or bk is None:
                    continue
                if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
                    continue
                if in_check(state, 'b' if state.turn == 'w' else 'w'):
                    continue

                positions.append(state)

            all_positions.extend(positions)
            self.filter_stats['source_positions_generated'] += len(positions)

        print(f"Generated {len(all_positions)} source positions from {len(self.source_materials)} materials")
        return all_positions

    def summary(self) -> str:
        """Get summary of filter statistics"""
        return (f"TargetedChessGame({self.target_material}):\n"
                f"  Target material found: {self.filter_stats['target_material_found']}\n"
                f"  Wrong material filtered: {self.filter_stats['wrong_material_filtered']}\n"
                f"  Source positions generated: {self.filter_stats['source_positions_generated']}")


def create_targeted_solver(target_material: str = "KQRRvKQR",
                           syzygy_path: str = "./syzygy",
                           max_memory_mb: int = 4000):
    """
    Create a HOLOS solver for targeted material search.

    Returns solver and game configured to find all 8-piece positions
    leading to the target 7-piece material.
    """
    from holos.holos import HOLOSSolver

    game = TargetedChessGame(syzygy_path, target_material)
    solver = HOLOSSolver(game, name=f"targeted_{target_material}", max_memory_mb=max_memory_mb)

    return solver, game
