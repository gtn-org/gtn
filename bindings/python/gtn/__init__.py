#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import subprocess
import tempfile

from .graph import *
from .graph import __version__
from .device import *
from .functions import *
from .autograd import *
from .creations import *
from .utils import *
from .rand import *
from .parallel import *
from . import cuda

def draw(graph, file_name, isymbols={}, osymbols={}):
    ext = os.path.splitext(file_name)[1]
    with tempfile.NamedTemporaryFile() as tmpf:
        write_dot(graph, tmpf.name, isymbols, osymbols)
        subprocess.check_call(["dot", "-T" + ext[1:], tmpf.name, "-o", file_name])
