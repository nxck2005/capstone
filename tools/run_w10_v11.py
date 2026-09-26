#!/usr/bin/env python3
"""Run the corrected W10 suffix under its v11 source and authority."""

from run_w10_v10 import main


if __name__ == "__main__":
    raise SystemExit(main(epoch="v11"))
