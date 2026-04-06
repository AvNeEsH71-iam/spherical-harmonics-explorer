"""
main.py
=======
Entry point for the Spherical Harmonics Explorer.

Usage
-----
    python main.py

Requirements
------------
    numpy >= 1.21
    scipy >= 1.7
    matplotlib >= 3.5
    tkinter  (ships with most Python distributions)
"""

import sys
import tkinter as tk


def main():
    # Friendly error if tkinter is missing (common on some Linux distros)
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"[ERROR] Could not open a Tk window: {exc}")
        print("  On Ubuntu/Debian, run:  sudo apt-get install python3-tk")
        sys.exit(1)

    # Import here so import errors are caught cleanly.
    # Support both package layout (explorer.app) and flat layout (app.py).
    try:
        try:
            from explorer.app import SphericalHarmonicsApp
        except ImportError:
            from app import SphericalHarmonicsApp
    except ImportError as exc:
        import tkinter.messagebox as mb
        mb.showerror("Import error",
                     f"A required package is missing:\n\n{exc}\n\n"
                     "Run:  pip install -r requirements.txt")
        sys.exit(1)

    app = SphericalHarmonicsApp(root)   # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
