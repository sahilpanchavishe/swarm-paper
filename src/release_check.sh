#!/bin/bash
set -e
cd ~/Developer/swarm-paper && source .venv/bin/activate
python src/check_claims.py
(cd src && python d8_check_numbers.py | tail -2 && python d9_check_numbers.py | tail -2)
git status --short | grep -v '^??' && { echo "UNCOMMITTED CHANGES: commit first"; exit 1; } || true
rm -rf /tmp/swarm-clone && git clone -q https://github.com/sahilpanchavishe/swarm-paper.git /tmp/swarm-clone
cd /tmp/swarm-clone && python src/d15_tables.py | tail -3 && python src/check_claims.py
echo "RELEASE CHECK PASSED. Freeze the text, then: git tag v1.0 && git push --tags"
