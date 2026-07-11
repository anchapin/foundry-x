"""CLI driver for text statistics.

The driver imports :func:`summarize` from the library and prints a
summary for a fixed sample string.  It is the entry point the agent
can run to verify the end-to-end integration.

This file is **not** edited in the gold path; it exists so the agent
must read across multiple files to understand the library's public
contract.
"""

from text_stats import summarize


def main() -> None:
    """Print a summary of a sample string."""
    sample = "the quick brown fox"
    print(summarize(sample))


if __name__ == "__main__":
    main()
