# Infosphere Phase Transition for ETFs

Treats the ensemble of market participants as an "infosphere" undergoing phase transitions. Information-theoretic order parameters (mutual information between agents, Fisher information of the collective state) signal imminent transitions. The per‑ETF score is the phase transition order parameter.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Mutual information between time-lagged agents (infosphere coherence)
- Fisher information of collective state (sensitivity)
- Phase order = geometric mean of coherence and sensitivity
- Score = order parameter (higher = nearing criticality)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-infosphere-phase-transition-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High phase order → nearing criticality / regime shift.
- Low phase order → stable regime.

## Requirements

See `requirements.txt`.
