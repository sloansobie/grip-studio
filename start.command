#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create the Python environment first; see README.md.'
  exit 1
fi
.venv/bin/python app.py
