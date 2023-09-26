# -*- coding: utf-8 -*-
# Created by Robert at 25.08.2023

import sys


def console_progressbar(count, total, suffix=''):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    # sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.write('\r[{}] {:5.1f}% ... {}'.format(bar, percents, suffix))
    if count == total:
        sys.stdout.write('\n')
    sys.stdout.flush()  # As suggested by Rom Ruben


def get_col_idx_from_header(header, names):
    """Get the column index from the column name(s)."""

    if not isinstance(names, list):
        names = [names]

    col_idx = []
    for name in names:
        col_idx.append(header.index(name))

    return col_idx
