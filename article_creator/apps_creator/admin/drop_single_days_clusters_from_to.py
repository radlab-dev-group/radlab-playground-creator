import os
import logging
import datetime
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.controllers.news import SummaryOfDayNewsController


def main(argv=None):
    # begin_end_date = datetime.date(2025, 5, 22)
    begin_end_date = None
    if begin_end_date is None:
        raise Exception("Begin date is not set!")

    end_time_delta = 1
    max_days_to_prepare = 60

    news_summary = SummaryOfDayNewsController()

    while True:
        if end_time_delta >= max_days_to_prepare:
            break

        end_date = begin_end_date - datetime.timedelta(days=end_time_delta)
        begin_date = end_date - datetime.timedelta(days=1)
        logging.info(
            f"Deleting informations in date range {begin_date} - {end_date}"
        )

        # Gen news for date range
        summaries = news_summary.get_summaries_for_day(date=begin_date)

        print("Number of summaries to delete: ", len(summaries))
        for summary in summaries:
            news_summary.force_drop_summary(summary=summary)

        end_time_delta += 1


if __name__ == "__main__":
    main()
