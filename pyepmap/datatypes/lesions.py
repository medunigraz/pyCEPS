# -*- coding: utf-8 -*-
# Created by Robert at 23.08.2023

from collections import namedtuple

Lesion = namedtuple('Lesion', ['X', 'diameter', 'RFIndex'])

RFIndex = namedtuple('RFIndex', ['name', 'value'])
