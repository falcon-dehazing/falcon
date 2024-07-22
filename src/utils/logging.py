import logging
import os
import os.path as osp


def setup_logging(log_directory):
    if not osp.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(filename=osp.join(log_directory, 'yaml_log.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')