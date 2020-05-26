#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:10:53 2020

@author: Sevans
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name     = 'PythonQOL',
      version  = '0.1',
      author   = 'Samuel Evans',
      author_email = 'sevans7@bu.edu',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url      = 'https://github.com/Sevans711/PythonQOL',
      packages = find_packages(),
      python_requires >= 3.0
      )