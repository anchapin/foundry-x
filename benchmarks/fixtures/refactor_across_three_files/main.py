"""Main entry point for refactor_across_three_files benchmark.

Runs processor.main() and server.main() sequentially so the benchmark
verifies both consumers of the constant are working after the rename.
"""

from processor import main as processor_main
from server import main as server_main

if __name__ == "__main__":
    processor_main()
    server_main()
