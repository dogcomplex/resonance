"""
Debug test to understand connection detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess import ChessGame, ChessState, random_position

def test_connection_detection():
    """Test that connections are detected when waves meet"""
    print("=" * 60)
    print("CONNECTION DETECTION TEST")
    print("=" * 60)

    # Create game with small piece count
    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="connection_test", max_memory_mb=500)

    # Create a simple 7-piece position (one above boundary)
    pos = random_position("KQRvKQR")  # 7 pieces
    if not pos:
        print("Could not generate position")
        return

    print(f"\nTest position ({pos.piece_count()} pieces):")
    pos.display()

    # Check if it's near boundary
    print(f"\nIs boundary (<=6 pieces): {game.is_boundary(pos)}")

    # Run a few iterations
    seeds = [SeedPoint(pos, SearchMode.WAVE)]

    print("\nRunning solver with verbose output...")
    print("-" * 40)

    # Manually run a few iterations to see what happens
    from holos.holos import LightningProbe

    # Initialize
    h = game.hash_state(pos)
    solver.forward_seen.add(h)
    solver.forward_frontier[h] = pos

    # Generate backward seeds from boundary
    generated = game.generate_boundary_seeds(pos, count=50)
    print(f"Generated {len(generated)} backward seeds")

    seeded = 0
    for seed in generated:
        sh = game.hash_state(seed)
        if sh not in solver.backward_seen:
            solver.backward_seen.add(sh)
            solver.backward_frontier[sh] = seed
            if game.is_boundary(seed):
                value = game.get_boundary_value(seed)
                if value is not None:
                    solver.solved[sh] = value
                    seeded += 1

    print(f"Seeded {seeded} backward positions with values")
    print(f"Forward frontier: {len(solver.forward_frontier)}")
    print(f"Backward frontier: {len(solver.backward_frontier)}")

    # Run iterations
    for i in range(5):
        print(f"\n--- Iteration {i} ---")

        fwd_contacts = solver._expand_forward()
        bwd_contacts = solver._expand_backward()

        # Check overlap
        overlap = solver.forward_seen & solver.backward_seen

        print(f"  Forward seen: {len(solver.forward_seen)}")
        print(f"  Backward seen: {len(solver.backward_seen)}")
        print(f"  Overlap: {len(overlap)}")
        print(f"  Solved: {len(solver.solved)}")
        print(f"  Forward contacts: {fwd_contacts}")
        print(f"  Backward contacts: {bwd_contacts}")
        print(f"  Stats connections: {solver.stats['connections']}")

        # Check for connections
        new_conns = solver._find_connections()
        if new_conns:
            print(f"  ** {new_conns} new connections found! **")

        # Stop if we have connections
        if solver.stats['connections'] > 0:
            print("\nConnections detected!")
            break

        if not solver.forward_frontier and not solver.backward_frontier:
            print("\nBoth frontiers empty!")
            break

    print("\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"Forward seen: {len(solver.forward_seen)}")
    print(f"Backward seen: {len(solver.backward_seen)}")
    print(f"Overlap: {len(solver.forward_seen & solver.backward_seen)}")
    print(f"Solved: {len(solver.solved)}")
    print(f"Connections (stats): {solver.stats['connections']}")
    print(f"Connections (list): {len(solver.connections)}")


def test_simple_meeting():
    """Even simpler test - start from boundary position"""
    print("\n" + "=" * 60)
    print("SIMPLE MEETING TEST")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)

    # Generate a 6-piece boundary position
    template = random_position("KQRvKQR")
    if not template:
        print("Could not generate template")
        return

    boundaries = game.generate_boundary_seeds(template, count=10)
    if not boundaries:
        print("Could not generate boundary positions")
        return

    boundary_pos = boundaries[0]
    print(f"Boundary position ({boundary_pos.piece_count()} pieces):")
    boundary_pos.display()

    # Get one predecessor (going backwards = increasing pieces)
    preds = list(game.get_predecessors(boundary_pos))
    if not preds:
        print("No predecessors!")
        return

    pred, move = preds[0]
    print(f"\nPredecessor ({pred.piece_count()} pieces):")
    pred.display()

    # Now: forward wave starts at pred, backward wave starts at boundary
    # They should meet at... the boundary position!

    solver = HOLOSSolver(game, name="simple_test")

    # Forward seed: the predecessor
    fh = game.hash_state(pred)
    solver.forward_seen.add(fh)
    solver.forward_frontier[fh] = pred

    # Backward seed: the boundary position
    bh = game.hash_state(boundary_pos)
    solver.backward_seen.add(bh)
    solver.backward_frontier[bh] = boundary_pos
    value = game.get_boundary_value(boundary_pos)
    if value is not None:
        solver.solved[bh] = value
        print(f"Boundary value: {value}")

    print(f"\nInitial state:")
    print(f"  Forward frontier: {len(solver.forward_frontier)} (starting at {pred.piece_count()}-piece)")
    print(f"  Backward frontier: {len(solver.backward_frontier)} (starting at {boundary_pos.piece_count()}-piece)")

    # One forward expansion should reach boundary
    print(f"\nExpanding forward...")
    fwd_contacts = solver._expand_forward()

    print(f"  Forward seen after: {len(solver.forward_seen)}")
    print(f"  Forward frontier after: {len(solver.forward_frontier)}")
    print(f"  Contacts: {fwd_contacts}")
    print(f"  Solved: {len(solver.solved)}")

    # Check if boundary position is now in forward_seen
    if bh in solver.forward_seen:
        print(f"  ** Boundary position reached by forward wave! **")

    overlap = solver.forward_seen & solver.backward_seen
    print(f"  Overlap: {len(overlap)}")

    # One backward expansion
    print(f"\nExpanding backward...")
    bwd_contacts = solver._expand_backward()

    print(f"  Backward seen after: {len(solver.backward_seen)}")
    print(f"  Backward frontier after: {len(solver.backward_frontier)}")
    print(f"  Contacts: {bwd_contacts}")
    print(f"  Stats connections: {solver.stats['connections']}")

    overlap = solver.forward_seen & solver.backward_seen
    print(f"  Overlap: {len(overlap)}")


if __name__ == "__main__":
    test_simple_meeting()
    test_connection_detection()
