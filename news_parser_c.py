import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../IB/ib_stocks_app')

import cfg
import proj_utils
import proj_ibtws_c
import threading
import proj_sec_scraping_utils
import proj_snapshotXmlReader
import proj_finStatementsXmlReader
import proj_sics_handler_c
from cfg import logging

YES = "y"
NO  = "n"





if __name__ == "__main__":

    host1 = "192.168.0.152"
    ib_api = proj_ibtws_c.IbTws(host = host1,
                           port = 7496,
                           clientId = 10)
    ib_api_run_thread = threading.Thread(target = ib_api.run)
    ib_api_run_thread.start()
    cfg.time.sleep(3)
    ib_api.call_function(cfg.GET_HISTORICAL_NEWS, ticker = "TEVA", startDate = "12/19/2021", endDate = "12/12/2020")
    # print("+".join([str(i.code) for i in ib_api.get_news_providers()]))

