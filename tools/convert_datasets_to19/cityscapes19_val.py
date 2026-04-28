"""Generate Cityscapes 19-class validation labels.

This is a small explicit entry point for users preparing the
``cityscapes19_val`` evaluation split. It delegates to ``cityscapes.py`` and
defaults to converting only the validation split.
"""

import sys

import cityscapes


def main():
    argv = sys.argv[1:]
    if not any(arg == '--splits' or arg.startswith('--splits=') for arg in argv):
        argv.extend(['--splits', 'val'])
    sys.argv = [sys.argv[0], *argv]
    cityscapes.main()


if __name__ == '__main__':
    main()
