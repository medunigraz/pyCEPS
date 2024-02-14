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

"""
Provides a class for IGB IO to/from python numpy arrays.
"""

import numpy as np
import gzip
import sys

FORMFEED = b'\x0c'
MAXLENGTH = 128*1024

# Save original file open
fopen = open

class IGBFile:
    """
    IGB format file IO class.

    Direct read/write to gzipped files is possible. The class automatically
    determines if the file is gzipped by its extension.
    """

    def __init__(self, filename, mode='r'):
        """
        Args:
            filename: (str) the filename to open.
        Kwargs:
            mode: (str, optional) <br>'r' for read mode (default) <br>'w' for write mode
        """

        # Make binary by default
        mode = {'r':'rb', 'w':'wb'}.get(mode, mode)

        if filename.endswith('.gz'):
            self._fp = gzip.open(filename, mode)
        else:
            self._fp = fopen(filename, mode)

        self._hdr_len = None
        self._hdr_content = None

    def close(self):
        """
        Close the file object.
        """
        self._fp.close()

    def _header_length(self):
        """
        Determine the length of the IGB file header, in bytes.

        Only relevant for read-mode files.

        Returns:
            (int) Number of bytes in the header
        """

        if self._hdr_len is not None:
            return self._hdr_len

        # Make sure at start
        self._fp.seek(0, 0)

        # Get first byte
        count = 1
        byte = self._fp.read(1)

        while byte != FORMFEED:

            # Check EOF not reached
            if byte == '':
                raise Exception('File ended before header read')

            # Check we are not accumulating unreasonably large header
            if count > MAXLENGTH:
                raise Exception('Header exceeds {0} bytes'.format(MAXLENGTH))

            # Read next byte
            byte = self._fp.read(1)
            count += 1

        # Cache result
        self._hdr_len = count

        return count

    def header(self):
        """
        Read the IGB file header and return as python dictionary.

        Returns:
            (dict) The contents of the file header
        """

        if self._hdr_content is not None:
            return self._hdr_content

        # Get header length
        hdr_len = self._header_length()

        # Rewind file
        self._fp.seek(0, 0)
        hdr_str = self._fp.read(hdr_len - 1)

        # Python 3 compatibility
        hdr_str = hdr_str.decode('utf-8')

        # Clean newline characters
        hdr_str = hdr_str.replace('\r', ' ')
        hdr_str = hdr_str.replace('\n', ' ')
        hdr_str = hdr_str.replace('\0', ' ')

        # Build dictionary of header content
        self._hdr_content = {}

        for part in hdr_str.split():

            key, value = part.split(':')

            if key in ['x', 'y', 'z', 't', 'bin', 'num', 'lut', 'comp']:
                self._hdr_content[key] = int(value)

            elif (key in ['facteur', 'zero', 'epais']
                  or key.startswith('org_')
                  or key.startswith('dim_')
                  or key.startswith('inc_')):
                self._hdr_content[key] = float(value)

            else:
                self._hdr_content[key] = value

        if 'inc_t' not in self._hdr_content:
            try:
                dim_t = self._hdr_content['dim_t']
                t = self._hdr_content['t']
            except KeyError:
                pass
            else:
                if t > 1:
                    self._hdr_content['inc_t'] = dim_t / (t - 1)

        return self._hdr_content

    def dtype(self):
        """
        Get a numpy-friendly data type for this file.

        Returns:
            (numpy.dtype) The numpy data type corresponding to the file contents
        """

        hdr = self.header()

        # Get numpy data type
        dtype = {'char':   np.byte,
                 'short':  np.short,
                 'int':    np.intc,
                 'long':   np.int_,
                 'ushort': np.ushort,
                 'uint':   np.uintc,
                 'float':  np.single,
                 'vec3f':  np.single,
                 'vec9f':  np.single,
                 'double': np.double}[hdr['type']]
        dtype = np.dtype(dtype)

        # Get python byte order string
        endian = {'little_endian': '<',
                  'big_endian':    '>'}[hdr['systeme']]

        # Return data type with correct order
        return dtype.newbyteorder(endian)

    def data(self):
        """
        Return a numpy array of the file contents.

        The data is returned as a flat array. It is up to the user to use the
        header information to determine how to reshape the array, if desired.

        Returns:
            (numpy.ndarray) A numpy array with the file contents
        """

        # Sanity check
        assert self.header()['type'] in ['int','float', 'vec3f', 'vec9f'], \
            'Only int, float, vec3f and vec9f currently supported'

        # Move to start of content
        hdr_len = self._header_length()
        self._fp.seek(hdr_len, 0)

        if isinstance(self._fp, gzip.GzipFile):
            # Read remaining file
            byte_str = self._fp.read()
            # Create a numpy array view on content
            data = np.frombuffer(byte_str, dtype=self.dtype())

        else:
            # Use more efficient direct read from file
            # This function uses the underlying C FILE pointer directly
            data = np.fromfile(self._fp, dtype=self.dtype())

        return data

    def write(self, data, header={}):
        """
        Write a numpy array to the IGB file.

        Some header fields are automatically determined from the numpy array.
        Others must be specified as additional keyword arguments.

        Not all IGB output formats are currently supported. For that reason,
        the input numpy array data type must be single.

        Args:
            data: (numpy.ndarray) The array to be written to the IGB file
        Kwargs:
            header: (dict) Fields to be included in the IGB header
        """

        # Sanity check
        assert data.dtype == np.single, \
            'Only float (single), vec3f and vec9f currently supported'

        # Compute length of data
        x   = data.size
        dim = 1
        t   = 1
        if 't' in header:
            t = int(header['t'])
            x /= t
        if 'type' in header:
            if header['type'] == 'vec3f':
                dim = 3
                x /= 3
            elif header['type'] == 'vec9f':
                dim = 9
                x /= 9
        if 'x' in header:
            x = header['x']

        # Check product makes sense
        assert x * t * dim == data.size, \
            'Provided header info does not match data size'

        # Construct header string
        hdr = {'x': x, 'y': 1, 'z': 1, 't': t, 'facteur': 1, 'zero': 0}
        hdr['type'] = {1: 'float', 3: 'vec3f', 9: 'vec9f'}[dim]
        hdr.update(header)

        # Set endianness correctly
        if data.dtype.byteorder in ['|', '=']:
            if sys.byteorder == 'little':
                endian = 'little_endian'
            else:
                endian = 'big_endian'
        elif data.dtype.byteorder == '<':
            endian = 'little_endian'
        elif data.dtype.byteorder == '>':
            endian = 'big_endian'
        hdr['systeme'] = endian

        # Write in preferred order
        hdr_entries = []
        for key in ['x', 'y', 'z', 't', 'type', 'systeme']:
            hdr_entries.append('{}:{} '.format(key, hdr[key]))
            del hdr[key]

        # Write remaining
        for key, value in hdr.items():
            hdr_entries.append('{}:{} '.format(key, value))

        lines = ['']
        for entry in hdr_entries:

            if len(lines[-1]) + len(entry) > 70:
                # Push new empty line on end
                lines.append('')

            lines[-1] += entry

        # Build header string
        hdr_str = '\r\n'.join(lines)
        # Terminate final line
        hdr_str += '\r\n'

        # Number of 1024-char blocks
        blocks = int(np.ceil(len(hdr_str) / 1024.))
        # Additional characters needed to complete blocks
        extra_chars = blocks * 1024 - len(hdr_str)

        # Add extra lines
        full_line_len = 72
        final_len = 3
        while extra_chars > (full_line_len + final_len):
            hdr_str += ' '*70 + '\r\n'
            extra_chars -= 72

        # Add final line
        spaces = extra_chars - final_len
        hdr_str += ' ' * spaces
        hdr_str += '\r\n'

        # Convert to bytes
        hdr_str = hdr_str.encode('utf-8')

        # Terminate header
        hdr_str += FORMFEED

        # Write header
        self._fp.seek(0, 0)
        byte = self._fp.write(hdr_str)

        # Write data to file
        if isinstance(self._fp, gzip.GzipFile):
            # Write from numpy array buffer
            self._fp.write(data.data)
        else:
            data.tofile(self._fp)

def open(*args, **kwargs):
    """
    Open an IGB file.

    Convenience method to provide normal python style interface to create a
    file type object.

    Args:
        filename: (str) The filename to open.
    Kwargs:
        mode: (str) 'r' for read mode (default), 'w' for write mode
    """
    return IGBFile(*args, **kwargs)

def read(filename, reshape=True):
    """
    Read an IGB file and return the reshaped data array plus header.

    Args:
        filename: (str) The filename to open
        reshape: (bool) Reshape the data array. Default: True.<br>DOFs are stored in the lines, time in the columns.
    Returns:
         data:   (numpy.ndarray) IGB data.
         header: (dict) Header information of IGB file.
         t:      (numpy.ndarray) Vector with time values associated with data points.
    """
    igbobj = IGBFile(filename)
    header = igbobj.header()
    data   = igbobj.data()
    igbobj.close()

    # reshape data
    num_traces = header.get('x')
    num_tsteps = header.get('t')
    dim_t      = header.get('dim_t')
    inc_t      = 1 if not header.get('inc_t') else header.get('inc_t')
    org_t      = 0 if not header.get('org_t') else header.get('org_t')  # origin in time is non-zero after restart

    if reshape == True:
        # allow reading data from unfinished simulations
        if len(data) / num_traces < num_tsteps:
            num_tsteps = int(len(data) / num_traces)
            dim_t      = (num_tsteps - 1) * inc_t
            data       = np.resize(data, num_tsteps * num_traces)

        data = data.reshape(num_tsteps, num_traces).T

    t    = np.linspace(0, dim_t, num_tsteps) + org_t
    ts   = inc_t                                # sampling time
    fs   = 1000. / ts                           # sampling frequency in Hz
    fnyq = fs / 2.                              # Nyquist frequency

    return data, header, t
