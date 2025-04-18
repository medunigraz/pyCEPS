# -*- coding: utf-8 -*-

# pyCEPS allows to import, visualize and translate clinical EAM data.
#     Copyright (C) 2023  Robert Arnold
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from typing import List, Union


def console_progressbar(
        count: int,
        total: int,
        suffix: str = ''
) -> None:
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


def get_col_idx_from_header(
        header,
        names: Union[str, List[str]]
) -> List[int]:
    """Get the column index from the column name(s)."""

    if not isinstance(names, list):
        names = [names]

    col_idx = []
    for name in names:
        col_idx.append(header.index(name))

    return col_idx
