#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for reading configurations.

Author: Jan Schlüter
"""

import io


def parse_variable_assignments(assignments):
    """
    Parses a list of key=value strings and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    """
    variables = {}
    for assignment in assignments or ():
        key, value = assignment.split('=', 1)
        for convert in (int, float, str):
            try:
                value = convert(value)
            except ValueError:
                continue
            else:
                break
        variables[key] = value
    return variables


def parse_config_file(filename):
    """
    Parses a file of key=value lines and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    Empty lines and lines starting with '#' are ignored.
    """
    with io.open(filename, 'r') as f:
        return parse_variable_assignments(
                [l.rstrip('\r\n') for l in f
                 if l.rstrip('\r\n') and not l.startswith('#')])
