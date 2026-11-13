GROUP54_CENG313: Forecasting Sticker Sales

Group ID: 54

Member: Bekir Esen, 23118080701

Competition: https://www.kaggle.com/competitions/playground-series-s5e1

This project is for the Kaggle Playground competition "Forecasting Sticker Sales". The goal is to predict how many stickers are sold each day in 6 countries, 3 stores and 5 products for 2017 to 2019, using daily sales from 2010 to 2016. Predictions are scored with MAPE (mean absolute percentage error), so lower is better. The competition is closed, so I evaluated my model with a late submission.

For Assignment 1 I explored the data and built a baseline. Country turned out to be the biggest factor by far (Kenya sells around 20 stickers a day, Norway around 1,400), followed by product and store, and weekends sell more than weekdays. The missing sales values are all in Canada and Kenya, mostly for the lowest-selling product, so I treated them as tiny unrecorded sales and dropped them. The baseline is a LightGBM model trained on calendar features and the category columns, using a log-transformed target so that its errors are relative like MAPE. I validated it by training on 2010 to 2015 and testing on 2016.

Results: on the 2016 validation year the model scores 6.94% MAPE, compared with 10.35% for a naive "last year's average" baseline. Without the log transform, the same model scores 11.50%, which is worse than the naive guess. The Kaggle late submission scores 12.55% public and 16.36% private. These are worse than validation because the test period is three years ahead, and tree models cannot extrapolate the yearly level shifts in the data. A screenshot of the score is in submissions/kaggle_score.png.

To get the data, accept the competition rules on the Kaggle page (Late Submission), download everything from the Data tab, and unzip train.csv and test.csv into a folder called data in the repository root. The data folder is not committed.

To run the project, create a virtual environment, install the packages and open the notebook from the repository root:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    jupyter notebook assignment1_eda_baseline.ipynb

Then run all cells. The notebook writes submissions/submission.csv, which is the file I uploaded to Kaggle. The reusable code is in the src folder: data.py loads and prepares the data, models.py has the baselines and the LightGBM model, and utils.py has the MAPE metric and the submission writer.

For Assignment 2 I plan to add GDP per capita for each country and year, which should explain the yearly level shifts, and holiday features for the recurring sales spikes.

This project was developed with generative AI assistance for the code and the write-up. The decisions, runs and results are my own, and I can explain every cell.
