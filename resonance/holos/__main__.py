"""
Allow running HOLOS as a module: python -m holos <command>
"""
from .cli import main

if __name__ == "__main__":
    main()
