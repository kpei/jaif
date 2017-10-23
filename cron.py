def main():
    import logging as logger
    import datetime as dt
    import sys

    # Set exception logging for uncaught ones
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    # Create logging
    log_filename = dt.date.today().strftime("%m-%d-%Y-CRONLOG.log")
    logger.basicConfig(filename="jaif/cron_logs/"+log_filename,level=logger.DEBUG)

    from pc import pull,calculateBenchmarkPrices,calculatePortfolios

    pull()
    calculateBenchmarkPrices()
    calculatePortfolios()

if __name__ == "__main__":
    main()

