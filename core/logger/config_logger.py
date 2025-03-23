import logging
import logging.handlers
from logging.config import dictConfig

# Logger chính của module
logger = logging.getLogger(__name__)

# Cấu hình ghi log mặc định
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
}
def configure_logging(logfile_path):
    """
    Initialize logging defaults for Project.

    :param logfile_path: logfile used to the logfile
    :type logfile_path: string

    This function does:

    - Assign INFO and DEBUG level to logger file handler and console handler

    """
    # Áp dụng cấu hình mặc định
    dictConfig(DEFAULT_LOGGING)

    # Định dạng thông tin log
    default_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)s] [PID:%(process)d TID:%(thread)d]\n %(message)s",
        "%d/%m/%Y %H:%M:%S")

    # Thiết lập xử lý file log với khả năng xoay vòng
    file_handler = logging.handlers.RotatingFileHandler(logfile_path, maxBytes=10485760,backupCount=300, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

# Thiết lập logging cơ bản với chỉ định file
def logging_without(logfile):
    # Cấu hình logging cơ bản
    logging.basicConfig(filename=logfile, filemode="w", level=logging.INFO)
    logging.captureWarnings(capture=False)
    
    # Vô hiệu hóa các logger không liên quan
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if "core" not in log_name:
            if log_name != "__main__":
                log_obj.disabled = True
        if "androguard" in log_name:
            log_obj.disabled = True