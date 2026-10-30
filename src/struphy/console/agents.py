import importlib.metadata
import shutil
from pathlib import Path

import struphy

GUIDES = Path(struphy.__path__[0]) / "agent_guides"

# marks the pointer block written into a project's AGENTS.md/CLAUDE.md, so that --install is idempotent
MARKER = "<!-- struphy agents -->"


def examples_url() -> str:
    """URL of the struphy-examples repository at the tag of the installed Struphy version
    (the examples folder on devel when not installed)."""
    try:
        version = importlib.metadata.version("struphy")
    except importlib.metadata.PackageNotFoundError:
        return "https://github.com/struphy-hub/struphy/tree/devel/examples"
    return f"https://github.com/struphy-hub/struphy-examples/tree/v{version}"


def skill_names() -> list[str]:
    """Names of the skills shipped with Struphy."""
    return sorted(p.parent.name for p in (GUIDES / "skills").glob("*/SKILL.md"))


def pointer_block() -> str:
    """Instructions written into a project's AGENTS.md/CLAUDE.md by ``struphy agents --install``."""
    skills = ", ".join(skill_names())
    return (
        f"{MARKER}\n"
        "## Struphy\n\n"
        "This project uses Struphy. Before writing Struphy parameter files or post-processing scripts, "
        "run `struphy agents` and follow the usage guide it prints. "
        f"Skills for the details are in `.claude/skills/` ({skills}); "
        "refresh them with `struphy agents --install` after upgrading Struphy.\n"
    )


def struphy_agents(show_path: bool = False, install: str | None = None):
    """Print the usage guide for AI coding agents, or install its skills into a project.

    Parameters
    ----------
    show_path : bool
        Print only the folder that holds the guide and the skills.

    install : str, optional
        Project directory to install into: the skills are copied to ``<install>/.claude/skills/``
        (replacing older copies of the same skills), and a pointer to the guide is added to
        ``<install>/AGENTS.md`` and ``<install>/CLAUDE.md`` unless already there.
    """
    if show_path:
        print(GUIDES)
    elif install is not None:
        install_skills(Path(install))
    else:
        skills = [str(GUIDES / "skills" / name / "SKILL.md") for name in skill_names()]
        print(f"Struphy usage guide for AI coding agents: {GUIDES / 'USAGE.md'}")
        print(f"Skills: {', '.join(skills)}")
        print(f"Examples for this version: {examples_url()}\n")
        print((GUIDES / "USAGE.md").read_text())


def install_skills(project: Path):
    """Copy the skills into ``project/.claude/skills/`` and point the project's agent files to them."""
    if not project.is_dir():
        raise FileNotFoundError(f"{project.resolve()} is not a directory")
    if (project / "src" / "struphy" / "agent_guides").resolve() == GUIDES.resolve():
        # its .claude/skills/ link to the guides already, and its AGENTS.md holds the coding guidelines
        print(f"{project.resolve()} is the Struphy repository of this installation, nothing to install")
        return

    target = project / ".claude" / "skills"
    for name in skill_names():
        source = GUIDES / "skills" / name
        dest = target / name
        if dest.resolve() == source.resolve():
            # e.g. the symlinks in .claude/skills/ of the Struphy repository
            print(f"Skill {name} in {dest} already links to the installed version")
            continue
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)
            print(f"Replaced the existing skill {name} in {dest}")
        shutil.copytree(source, dest)
        print(f"Installed skill {name} in {dest}")

    # AGENTS.md is read by most agents; Claude Code reads CLAUDE.md, which can import AGENTS.md
    agents_md = project / "AGENTS.md"
    add_pointer(agents_md)
    claude_md = project / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text("@AGENTS.md\n")
        print(f"Created {claude_md} (imports AGENTS.md)")
    elif not any(line.strip() == "@AGENTS.md" for line in claude_md.read_text().splitlines()):
        add_pointer(claude_md)


def add_pointer(file: Path):
    """Append the pointer block to ``file`` (created if missing) unless it is already there."""
    text = file.read_text() if file.exists() else ""
    if MARKER in text:
        print(f"{file} already points to the Struphy guide")
        return
    if text:
        text = text.rstrip("\n") + "\n\n"
    file.write_text(text + pointer_block())
    print(f"Added a pointer to the Struphy guide to {file}")
