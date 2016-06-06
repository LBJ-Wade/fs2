import fs._fs as c

def set_loglevel(loglevel):
    """Set the amount of standard output messages.

    Args:
        loglevel (int): 0 -- 7
    
    * 0: msg_debug
    * 1: msg_verbose
    * 2: msg_info
    * 3: msg_warn
    * 4: msg_error
    * 5: msg_fatal
    """
    return c.set_loglevel(loglevel)
