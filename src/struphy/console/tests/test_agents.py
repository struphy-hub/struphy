"""Tests for ``struphy agents``."""

import pytest

from struphy.console.agents import GUIDES, MARKER, skill_names, struphy_agents

# console commands, no parallel code involved; tmp_path differs per rank
pytestmark = pytest.mark.mpi_skip


def test_guides_shipped():
    assert (GUIDES / "USAGE.md").is_file()
    names = skill_names()
    assert {"setup-simulation", "postprocess-simulation"} <= set(names)
    for name in names:
        # the skill frontmatter must name the skill like its folder
        text = (GUIDES / "skills" / name / "SKILL.md").read_text()
        assert text.startswith("---\n") and f"\nname: {name}\n" in text


def test_print_guide(capsys):
    struphy_agents()
    out = capsys.readouterr().out
    assert "USAGE.md" in out and "/examples" in out
    assert "# Using Struphy" in out

    struphy_agents(show_path=True)
    assert capsys.readouterr().out.strip() == str(GUIDES)


def test_install(tmp_path, capsys):
    # an existing CLAUDE.md without an AGENTS.md import gets its own pointer
    (tmp_path / "CLAUDE.md").write_text("# My project\n")

    for _ in range(2):  # installing twice refreshes the skills without duplicating the pointers
        struphy_agents(install=str(tmp_path))

    for name in skill_names():
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").is_file()
    assert (tmp_path / "AGENTS.md").read_text().count(MARKER) == 1
    claude_md = (tmp_path / "CLAUDE.md").read_text()
    assert claude_md.startswith("# My project\n\n") and claude_md.count(MARKER) == 1
    assert "already points" in capsys.readouterr().out


def test_install_creates_claude_md(tmp_path):
    struphy_agents(install=str(tmp_path))
    assert (tmp_path / "CLAUDE.md").read_text() == "@AGENTS.md\n"
    assert MARKER in (tmp_path / "AGENTS.md").read_text()

    with pytest.raises(FileNotFoundError):
        struphy_agents(install=str(tmp_path / "missing"))
