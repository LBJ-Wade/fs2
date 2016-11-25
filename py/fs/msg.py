import fs._fs as c

dict_loglevel = {'debug': 0, 'verbose': 1, 'info': 2, 'warn': 3, 'error': 4,
                 'fatal': 5}


def set_loglevel(loglevel):
    """Set the amount of standard output messages.

    Args:
        loglevel (int or string): 0 -- 7

      int / str
    * 0 'debug'  : msg_debug
    * 1 'verbose': msg_verbose
    * 2 'info'   : msg_info
    * 3 'warn'   : msg_warn
    * 4 'error'  : msg_error
    * 5 'fatal'  : msg_fatal
    """

    if isinstance(loglevel, str):
        loglevel = dict_loglevel[loglevel]

    return c.set_loglevel(loglevel)
