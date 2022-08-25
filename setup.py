"""
Empty setup.py file to enable editable installation.
Do not change. Use setup.cfg.
Based on https://github.com/cta-observatory/project-template-python-pure
"""

from setuptools import setup

# this is a workaround for an issue in pip that prevents editable installation
# with --user, see https://github.com/pypa/pip/issues/7953
import site
import sys


site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup()
