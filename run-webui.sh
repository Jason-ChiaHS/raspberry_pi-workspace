#!/bin/bash

npm run --prefix scripts/web_ui/ build
npm run --prefix scripts/web_ui/ preview -- --host&
./.venv/bin/python scripts/show_result_to_web_ui.py --log_level 1
