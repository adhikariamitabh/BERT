#!/bin/bash
cd "$(dirname "$0")"

# Open a new terminal tab/window, activate the env, and start Jupyter
osascript -e 'tell application "Terminal" to do script "cd '"$(pwd)"' && source bert-env/bin/activate && jupyter lab"'
