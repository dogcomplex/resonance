"""
holos/cli.py - Unified Command Line Interface for HOLOS

This is the main entry point for running HOLOS searches.

Usage:
    python -m holos.cli <command> [options]

Commands:
    test        Quick test run (few batches, verify setup)
    run         Full production run (all batches, disk-backed)
    status      Check progress of an existing search
    demo        Run demo on Connect-4 or Sudoku

Examples:
    # Test run (5 batches, ~5 minutes)
    python -m holos.cli test --target KQRRvKQR

    # Full run (all batches, hours/days)
    python -m holos.cli run --target KQRRvKQR

    # Resume interrupted run
    python -m holos.cli run --target KQRRvKQR --resume

    # Check status
    python -m holos.cli status --target KQRRvKQR

    # Run demo
    python -m holos.cli demo --game connect4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def cmd_test(args):
    """Quick test run to verify setup"""
    from holos.full_search import FullSearchSession

    print("=" * 70)
    print("HOLOS TEST RUN")
    print("=" * 70)
    print(f"Target: {args.target}")
    print(f"This will run {args.batches} batches to verify the system works.")
    print()

    session = FullSearchSession(
        target_material=args.target,
        save_dir=args.search_dir or f"./test_{args.target}",
        syzygy_path=args.syzygy,
        batch_size=50,  # Smaller batches for test
        subprocess_memory_mb=args.memory,
        subprocess_iterations=8,  # Fewer iterations for test
        min_disk_gb=1.0,  # Lower threshold for test
    )

    # Initialize if needed
    if not session.state.forward_seeds_file:
        session.initialize_seeds(
            backward_count=500,  # Fewer seeds for test
            forward_per_material=30,
        )

    # Run limited batches
    session.run(max_batches=args.batches)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {session.save_dir}")
    print(f"To run full search: python -m holos.cli run --target {args.target}")


def cmd_run(args):
    """Full production run"""
    from holos.full_search import FullSearchSession

    print("=" * 70)
    print("HOLOS FULL SEARCH")
    print("=" * 70)
    print(f"Target: {args.target}")
    print()

    session = FullSearchSession(
        target_material=args.target,
        save_dir=args.search_dir or f"./search_{args.target}",
        syzygy_path=args.syzygy,
        batch_size=args.batch_size,
        subprocess_memory_mb=args.memory,
        subprocess_iterations=args.iterations,
        min_disk_gb=args.min_disk_gb,
    )

    # Initialize if needed (or if --init flag)
    if args.init or not session.state.forward_seeds_file:
        session.initialize_seeds(
            backward_count=args.backward_seeds,
            forward_per_material=args.forward_seeds,
        )

    # Run
    session.run(max_batches=args.max_batches)


def cmd_status(args):
    """Check status of existing search"""
    from holos.full_search import FullSearchSession

    session = FullSearchSession(
        target_material=args.target,
        save_dir=args.search_dir or f"./search_{args.target}",
        syzygy_path=args.syzygy,
    )

    print("=" * 70)
    print(f"HOLOS SEARCH STATUS: {args.target}")
    print("=" * 70)
    print(session.state.summary())
    print()
    print(session.hologram.summary())

    # Resource check
    ok, msg = session.check_resources()
    print(f"\nResources: {msg}")


def cmd_demo(args):
    """Run demo on a simple game"""
    if args.game == "connect4":
        from holos.demo import demo_connect4
        demo_connect4()
    elif args.game == "sudoku":
        from holos.demo import demo_sudoku
        demo_sudoku()
    else:
        print(f"Unknown game: {args.game}")
        print("Available: connect4, sudoku")


def main():
    parser = argparse.ArgumentParser(
        description="HOLOS - Universal Bidirectional Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  test      Quick test run (~5 minutes)
  run       Full production run (hours/days)
  status    Check progress
  demo      Run demo on simple games

Examples:
  python -m holos.cli test --target KQRRvKQR
  python -m holos.cli run --target KQRRvKQR --memory 4000
  python -m holos.cli status --target KQRRvKQR
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # === TEST command ===
    test_parser = subparsers.add_parser("test", help="Quick test run")
    test_parser.add_argument("--target", default="KQRRvKQR",
                             help="Target material (default: KQRRvKQR)")
    test_parser.add_argument("--batches", type=int, default=5,
                             help="Number of test batches (default: 5)")
    test_parser.add_argument("--memory", type=int, default=2000,
                             help="Memory limit MB (default: 2000)")
    test_parser.add_argument("--syzygy", default="./syzygy",
                             help="Syzygy tablebase path")
    test_parser.add_argument("--search-dir", default=None,
                             help="Output directory")

    # === RUN command ===
    run_parser = subparsers.add_parser("run", help="Full production run")
    run_parser.add_argument("--target", default="KQRRvKQR",
                            help="Target material (default: KQRRvKQR)")
    run_parser.add_argument("--init", action="store_true",
                            help="Force re-initialization of seeds")
    run_parser.add_argument("--backward-seeds", type=int, default=5000,
                            help="Number of backward seeds (default: 5000)")
    run_parser.add_argument("--forward-seeds", type=int, default=200,
                            help="Forward seeds per material (default: 200)")
    run_parser.add_argument("--batch-size", type=int, default=100,
                            help="Seeds per batch (default: 100)")
    run_parser.add_argument("--iterations", type=int, default=12,
                            help="Iterations per batch (default: 12)")
    run_parser.add_argument("--memory", type=int, default=3000,
                            help="Memory limit MB per batch (default: 3000)")
    run_parser.add_argument("--max-batches", type=int, default=None,
                            help="Max batches to run (default: unlimited)")
    run_parser.add_argument("--min-disk-gb", type=float, default=10.0,
                            help="Minimum free disk space GB (default: 10)")
    run_parser.add_argument("--syzygy", default="./syzygy",
                            help="Syzygy tablebase path")
    run_parser.add_argument("--search-dir", default=None,
                            help="Output directory")

    # === STATUS command ===
    status_parser = subparsers.add_parser("status", help="Check search status")
    status_parser.add_argument("--target", default="KQRRvKQR",
                               help="Target material")
    status_parser.add_argument("--syzygy", default="./syzygy",
                               help="Syzygy tablebase path")
    status_parser.add_argument("--search-dir", default=None,
                               help="Search directory")

    # === DEMO command ===
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    demo_parser.add_argument("--game", default="connect4",
                             choices=["connect4", "sudoku"],
                             help="Game to demo (default: connect4)")

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
