#!/usr/bin/env bash
# env setup for the jetson run. safe to re run.
# jetson_clocks does not survive a reboot so run this again after every boot.
# power mode is left alone, box is already on MAXN_SUPER (mode 2).
set -euo pipefail

echo "== platform =="
cat /etc/nv_tegra_release 2>/dev/null || echo "no nv_tegra_release"
uname -m; python3 --version; nproc; free -h

echo "== clocks =="
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show | head -20

echo "== quiet the box =="
sudo systemctl stop unattended-upgrades 2>/dev/null || true
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true

echo "== venv =="
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "== import check =="
python -c "import numpy,scipy,sklearn,pysindy,joblib; \
print('numpy',numpy.__version__,'scipy',scipy.__version__, \
'sklearn',sklearn.__version__,'pysindy',pysindy.__version__,'joblib',joblib.__version__)"
echo "setup done. idle 60s before benchmarking."
