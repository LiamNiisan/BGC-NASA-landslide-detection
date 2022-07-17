import os
import config
from data import data
from models.baseline import baseline
from models.multitask import multitask
from models.multitask.multitask import LandslideEventsClassifier
from models.multitask.multitask import LandslideEventsLabelClassifier
from models.multitask.multitask import LandslideEventsSpanClassifier


def main():
    start_date, end_date = config.get_interval()
    articles_df = data.get_articles_df(start_date, end_date)

    if config.is_running_baseline():
        config.logger.info("baseline : predicting articles...")
        predictions = baseline.predict(articles_df)
        output_df = data.get_formatted_output_df(articles_df, predictions)
        config.logger.info("baseline : saving results...")
        output_df.to_csv(os.path.join(config.user_path, "baseline_results.csv"))
        config.logger.info(f"baseline : results saved to {config.user_path}")
    
    if config.is_running_multitask():
        config.logger.info("multitask : predicting articles...")
        predictions = multitask.predict(articles_df)
        output_df = data.get_formatted_output_df(articles_df, predictions)
        config.logger.info("multitask : saving results...")
        output_df.to_csv(os.path.join(config.user_path, "multitask_results.csv"))
        config.logger.info(f"multitask : results saved to {config.user_path}")


if __name__ == "__main__":
    main()
