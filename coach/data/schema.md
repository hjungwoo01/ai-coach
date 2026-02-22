# Data Schema

## `sample_players.csv`
Columns:
- `player_id`: canonical stable ID
- `name`: display name
- `country`: country code/name
- `handedness`: `R` or `L`

## `sample_matches.csv`
Each row is one completed match from the perspective of the recorded `playerA_id` vs `playerB_id`.

Columns:
- `date`: match date (`YYYY-MM-DD`)
- `playerA_id`, `playerB_id`: participants
- `winner_id`: winner player ID
- `a_games_won`, `b_games_won`: games won in best-of-3
- `a_points`, `b_points`: total rally points won in match
- `a_serve_rallies`, `a_serve_wins`: A serve opportunities and won rallies
- `b_serve_rallies`, `b_serve_wins`: B serve opportunities and won rallies
- `a_short_serve_rate`, `a_flick_serve_rate`: A serve mix (sums to ~1)
- `a_attack_rate`, `a_neutral_rate`, `a_safe_rate`: A rally style mix (sums to ~1)
- `b_short_serve_rate`, `b_flick_serve_rate`: B serve mix (sums to ~1)
- `b_attack_rate`, `b_neutral_rate`, `b_safe_rate`: B rally style mix (sums to ~1)

## Parameter Estimation Assumptions
- Base rally probabilities are estimated from historical serve/receive outcomes.
- Receive-win counts use opponent serve totals: `receive_wins = opp_serve_rallies - opp_serve_wins`.
- Laplace smoothing is applied to avoid 0/1 probabilities:
  - `p = (wins + alpha) / (trials + 2*alpha)`
- Strategy mixes are weighted averages with Dirichlet-style pseudocounts.
- Head-to-head is blended with player baseline (small-sample shrinkage).

## Limits
- Match-level proxies are used instead of full rally logs.
- Style effects (`w_short`, `w_attack`, `w_safe`) are estimated from historical aggregate trends.
