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


class MapAttributeError(Exception):
    """Exception raised if failed to retrieve map attributes from study.xml.

    Parameters:
        message : string
    """

    def __init__(
            self,
            message: str
    ) -> None:
        """Constructor."""
        self.message = message
        super().__init__(self.message)


class MeshFileNotFoundError(Exception):
    """Exception raised if not mesh file was found in study.

    Parameters:
        filename : string
        message : string
    """

    def __init__(
            self,
            filename: str,
            message: str = 'Mesh file {} not found!'
    ) -> None:
        """Constructor."""
        self.filename = filename
        self.message = message.format(self.filename)

        super().__init__(self.message)
