import fs._fs as c


def set_filename(filename):
    c._stat_set_filename(filename)


def record_pm_nbuf(istep):
    group = "step%d" % istep
    c._stat_record_pm_nbuf(group)
