Local touchdown modeling package.

Modules:
- td_likelihood: team-level anytime TD likelihoods from spread/total + priors + weather.
- player_td_likelihood: allocates team TDs to players using usage priors and rosters.
- data_sources/features/weather/priors/schemas/team_normalizer: support modules.

Data expected under ../data relative to this package:
- games.csv, team_stats.csv, lines.csv, stadium_meta.csv, game_location_overrides.csv
- player_usage_priors.csv
- optional real_betting_lines_*.json and weather_YYYY-MM-DD.csv files
