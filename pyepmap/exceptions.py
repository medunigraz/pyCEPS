# -*- coding: utf-8 -*-
# Created by Robert at 24.08.2023


class MapAttributeError(Exception):
    """Exception raised if failed to retrieve map attributes from study.xml.

    Parameters:
        message : string
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MeshFileNotFoundError(Exception):
    """Exception raised if not mesh file was found in study.

    Parameters:
        filename : string
        message : string
    """

    def __init__(self, filename, message='Mesh file {} not found!'):
        self.filename = filename
        self.message = message.format(self.filename)

        super().__init__(self.message)
