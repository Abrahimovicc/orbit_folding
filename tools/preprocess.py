#!/usr/bin/env python3
"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.

Standalone counterpart of the orbit preprocess command.
"""
import sys
from orbit_folding.cli import main

if __name__ == "__main__":
    main(["preprocess", *sys.argv[1:]])
