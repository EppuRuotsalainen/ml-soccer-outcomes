# ML Soccer Outcomes

## Setup
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
git clone https://github.com/xgabora/Club-Football-Match-Data-2000-2025 external/Club-Football-Match-Data-2000-2025
```

## Build the report
```bash
cd report
latexmk -pdf -shell-escape main.tex
```