# Changelog

## [1.0.1] - 2024-04-16

### Added

- interpolation of IMP and FRC surface parameter maps (Robert Arnold)

### Changed

- export point impedance and contact force data only if data is available (Robert Arnold)
- test data set (added point force and impedance data) (Robert Arnold)

### Fixed

- load .pyceps with no additional meshes (Robert Arnold)
- load .pyceps with maps without points (Robert Arnold)
- import of point impedance and contact force data (Robert Arnold)

## [1.0.0] - 2024-04-10

### Added

- Export of second unipolar EGM data and point cloud (Robert Arnold)
- Added CLI argument --password for protected archives (Robert Arnold)

### Changed

- Studies are saved in new .pyceps format (Robert Arnold)
- Dimension of coordinate triples is now (3, ) instead of (3, 1) (Robert Arnold)
- Renamed CLI command --study-pkl to --study-file (Robert Arnold)
- CLI command --system is required only when importing data from repository (Robert Arnold)
- Changed test data to new export format (Robert Arnold)
- Added additional Meshes to test data set (Robert Arnold)
- Renamed EPStudy attribute .ecg to .bsecg (Robert Arnold)
- Set second unipolar coordinates to recording position if not found to avoid NaN's in output files (Robert Arnold)

### Fixed

- Error when overwriting existing study file is declined (Robert Arnold)

_Old .pkl files are not supported anymore!_

## [0.1.1] - 2024-04-02

### Changed

- Renamed EPStudy function set_root to set_repository (Robert Arnold)

### Fixed

- Set proper study root if data repository is not reachable (Robert Arnold)
- Import only specific maps if given as CLI command (Robert Arnold)

## [0.1.0] - 2024-03-11

### Added

- Determine position of second channel for bipolar recordings (Robert Arnold)
- Export of additional point info NAME, WOI, REF (Robert Arnold)
- Added QMODE+ test data (Robert Arnold)

### Changed

- Compatibility with Python >=3.8 <= 3.12 (Robert Arnold)
- Export of VisiTag data separate for "normal" and QMODE+ (Robert Arnold)

### Fixed

- Import maps with zero points (Robert Arnold)
- Unintentionally skipped lines in VisiTag files (Robert Arnold)

## [0.0.3] - 2024-02-14

### Fixed

- Missing colormaps.json (Robert Arnold)

## [0.0.2] - 2024-02-14

### Added

- Provided artificial Carto3 data set for testing (Robert Arnold)

### Changed

- Renamed module from src to pyceps (Robert Arnold)

### Fixed

- Pass map name as list to import method (Robert Arnold)

_This release was yanked on PyPi due to a missing file._

## [0.0.1] - 2024-02-01

_Initial release._

[1.0.1]: https://github.com/medunigraz/pyCEPS/releases/tag/1.0.1
[1.0.0]: https://github.com/medunigraz/pyCEPS/releases/tag/1.0.0
[0.1.1]: https://github.com/medunigraz/pyCEPS/releases/tag/0.1.1
[0.1.0]: https://github.com/medunigraz/pyCEPS/releases/tag/0.1.0
[0.0.3]: https://github.com/medunigraz/pyCEPS/releases/tag/0.0.3
[0.0.2]: https://github.com/medunigraz/pyCEPS/releases/tag/0.0.2
[0.0.1]: https://github.com/medunigraz/pyCEPS/releases/tag/0.0.1