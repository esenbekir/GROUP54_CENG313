# GROUP[ID]_CENG313 — Forecasting Sticker Sales

| | |
|---|---|
| **Group ID** | [ID] |
| **Member** | [Name Surname] — [Student ID] |
| **Competition** | [Playground Series S5E1 – Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1) |
| **Status** | Competition closed — evaluated with a Kaggle late submission |

## Problem
Predict daily sticker sales (`num_sold`) for 2017–2019 from daily sales in 2010–2016, for 6 countries × 3 stores × 5 products (90 series). The metric is **MAPE** (mean absolute percentage error, lower is better).

## Scores
| Evaluation | MAPE |
|---|---|
| Validation, trained on 2010–2015, tested on 2016 | 6.94% |
| Kaggle late submission, public | [PUBLIC] |
| Kaggle late submission, private | [PRIVATE] |

Screenshot: [`submissions/kaggle_score.png`](submissions/kaggle_score.png)

## How to download the data
1. Accept the rules on the [competition page](https://www.kaggle.com/competitions/playground-series-s5e1) (**Late Submission**).
2. Open the **Data** tab and click **Download All**.
3. Unzip into a `data/` folder at the repository root, so you have `data/train.csv` and `data/test.csv`.

The `data/` folder is excluded by `.gitignore` and is not committed.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook assignment1_eda_baseline.ipynb
```
Run all cells from the repository root (**Restart & Run All**). The notebook writes `submissions/submission.csv`, which is the file uploaded to Kaggle.

## Repository structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA & baseline
├── src/
│   ├── __init__.py
│   ├── data.py      # load_data, preprocess
│   ├── models.py    # naive_baseline, train_model, predict
│   └── utils.py     # mape, save_submission
├── data/            # not committed
└── submissions/     # score screenshots
```

## Approach (Assignment 1)
- **EDA:** country is the strongest driver (Kenya ~20/day vs Norway ~1,423/day), then product and store; weekends sell more; missing targets are tiny unrecorded sales in Canada and Kenya.
- **Preprocessing:** drop missing targets, split the date into calendar features, mark country/store/product as categories.
- **Baseline:** LightGBM on `log1p(num_sold)` with a time-based validation split, compared against a naive "last year's average" baseline.

## Next steps (Assignment 2)
- GDP per capita per country and year, to capture year-level shifts the current model cannot extrapolate.
- Holiday features for the recurring sales spikes.
