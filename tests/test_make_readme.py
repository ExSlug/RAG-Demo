from src.make_readme import END, START, replace_block


def test_replace_block_changes_only_the_marked_part():
    readme = f"intro\n{START}\nold numbers\n{END}\noutro"
    new = replace_block(readme, f"{START}\nnew numbers\n{END}")
    assert new == f"intro\n{START}\nnew numbers\n{END}\noutro"
    assert "old numbers" not in new
