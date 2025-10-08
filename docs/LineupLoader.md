

**Overview**: Role inference, errors when no suspects provided/detected, default target, lineup generation rules, grid visualisation.


- `generate_lineup(lineup_size=None, target_lineup=None, shuffle=False)`:
  - If neither `lineup_size` nor `target_lineup` provided, uses **all images** in a combined lineup.
  - Else: guilty (TP) / innocent (TA) + fillers to match the desired size. 
  - Errors if not enough fillers.
- `show_lineup(...)`: draws a grid of lineup images, auto layouts, and optional role titles.
- **Edge cases**: insufficient fillers; duplicate images; 
explicit path overrides filename heuristics. 
