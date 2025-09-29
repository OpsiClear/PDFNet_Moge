"""
Common utility functions shared across PDFNet modules.

This module contains utility functions that are used in multiple places
to avoid code duplication.
"""

import os


def get_files(PATH):
    """
    Get all files in a directory recursively.

    Args:
        PATH: Root directory path(s) to search - can be a string or list of strings

    Returns:
        List of file paths found in the directory tree(s)
    """
    file_list = []
    if isinstance(PATH, str):
        for filepath, dirnames, filenames in os.walk(PATH):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
    elif isinstance(PATH, list):
        for path in PATH:
            for filepath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_list.append(os.path.join(filepath, filename))
    return file_list