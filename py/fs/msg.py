import fs._fs as c

dict_loglevel = {'debug': 0, 'verbose': 1, 'info': 2, 'warn': 3, 'error': 4,
                 'fatal': 5}


def set_loglevel(loglevel):
    """Set the amount of standard output messages.

    Args:
        loglevel (int or string):

    * 0 'debug'
    * 1 'verbose'
    * 2 'info'
    * 3 'warn'
    * 4 'error'
    * 5 'fatal'
    """

    if isinstance(loglevel, str):
        loglevel = dict_loglevel[loglevel]

    c.set_loglevel(loglevel)
