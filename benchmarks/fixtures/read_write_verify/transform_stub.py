"""Transform module for read_write_verify benchmark.

The transform_row() function is a stub: it raises NotImplementedError,
causing the first run to fail.  The golden fix implements the function
body, leaving this docstring and the if __name__ guard intact.
"""


def transform_row(row: dict) -> dict:
    """Transform a row: uppercase the name field and double the score field.

    Args:
        row: A dictionary with 'id', 'name', and 'score' keys.

    Returns:
        A new dictionary with 'id' unchanged, 'name' uppercased,
        and 'score' doubled.
    """
    raise NotImplementedError("transform_row not yet implemented")


if __name__ == "__main__":
    import csv

    with open("input.csv", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    transformed = [transform_row(row) for row in rows]

    with open("output.csv", "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["id", "name", "score"])
        writer.writeheader()
        writer.writerows(transformed)

    print(f"Processed {len(transformed)} rows.")
