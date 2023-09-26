# -*- coding: utf-8 -*-
# Created by Robert at 23.08.2023


from collections import namedtuple


Trace = namedtuple('Trace', ['name', 'data', 'fs'])

BodySurfaceECG = namedtuple('BodySurfaceECG',
                            ['method', 'refAnnotation', 'traces'])
