#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re

from pyceps.carto import CartoStudy, CartoMap
from pyceps.fileio.cartoio import read_ecg_file_header

logging.basicConfig(level=logging.INFO)


"""SPECIFY DATA LOCATION HERE"""
study_repo = (r'D:\pyCEPS_testdata\pyceps_allEGM\Export_FliFla-01_25_2022-09-46-01.zip')
pwd = ''
encoding = 'cp1252'
export_loc = r'D:\pyCEPS_testdata\pyceps_allEGM\test_export'


"""LOAD AND EXPORT EGM DATA"""
# import study
study = CartoStudy(study_repo, pwd=pwd, encoding=encoding)
study.import_study()
study.list_maps()

# import/export maps
map_names = study.mapNames  # adjust this as needed
for m in map_names:
    logging.info('working on map {}'.format(m))
    try:
        new_map = CartoMap(m, study.studyXML, parent=study)
        new_map.import_attributes()
        new_map.surface = new_map.load_mesh()  # needed for projected X
        new_map.points = new_map.load_points(
            study_tags=study.mappingParams.TagsTable,
            egm_names_from_pos=False
        )
        # get EGM names from 1st point
        p = new_map.points[0]
        p_ecg = study.repository.join(p.ecgFile)
        with study.repository.open(p_ecg) as fid:
            ecg_header = read_ecg_file_header(fid)
            ecg_names = [re.search(r'(.+?)\(', x).group(1)
                         for x in ecg_header['ecg_names']
                         ]
        new_map.export_point_ecg(output_folder=export_loc,
                                 which=ecg_names,
                                 reload_data=False
                                 )
    except Exception as err:
        logging.warning('failed to import map {}: {}'
                        .format(m, err))
        continue
