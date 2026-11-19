# Premier League Match Predictor

This project uses past Premier League results to predict a home win, a draw, or an away win. The notebook shows the most likely outcome and its estimated probability for each match.

## How to explain it

> I used Premier League results from 2010 to 2026 to build a match predictor. It looks at how strong each team is and how it has played recently. I tested it by going through past matches in order and making each prediction using only the results available before that match.

## How to run it

1. Open [Premier_League_Predictor.ipynb](Premier_League_Predictor.ipynb) and choose **Run All**. Keep `premier_league.csv` and `fixtures_2026_27.csv` beside the notebook.
2. Scroll to **Upcoming match picks** at the bottom. **Top pick** names a club or says **Draw**, with the probability beside it.
3. To try your own matchup, edit the team names and date under **Make a prediction**, then rerun that cell.

If you need to install the notebook tools, run these commands from the project folder with Python 3.12.

```sh
python -m pip install -r requirements-notebook.txt
python -m notebook Premier_League_Predictor.ipynb
```

## How it works

- Each club gets an **Elo rating**, a measure of team strength that updates after each result.
- The model combines those ratings with the teams' recent results, goals, and shots on target.
- It estimates three probabilities that add up to 100%. The highest one becomes the top pick.

## How well it did

On a historical check of **6,301 matches**, the model's top pick was right **53.4%** of the time. Pre-match bookmaker odds reached **54.5%** on the same matches. Draws were a weakness for the model. The live scoreboard tracks forecasts recorded before kickoff so future results can give a separate check.

## Where the data comes from

The local archive contains **6,364 completed matches** from January 2010 through November 8, 2026. The results and historical odds came from Football-Data.co.uk. The schedule comes from the Premier League. The provider describes its data as intended for league-match prediction. Check its terms before sharing the CSV. [Results source and terms](https://football-data.co.uk/data.php) · [Official fixture list](https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-season-202627-premier-league-season)
