"""
Module specifying the current version string for struphy.
"""

__version__ = "1.9.8"

def display_version():
    print(f'struphy {__version__}\n\
Copyright 2022 (c) struphy dev team | CONTRIBUTING.md | Max Planck Institute for Plasma Physics\n\
MIT license\n\
This is free software, no warranty.')

if __name__ == "__main__":
    display_version()
