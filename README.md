# pyCEPS

pyEPmap provides an interface to import, visualize and translate clinical mapping data.
Supported mapping systems are: CARTO<sup>&reg;</sup>3 (Biosense Webster), EnSite Precision<sup>&trade;</sup> (Abbot).

### Installation
Python 3.6 or higher is required. Just use [pip](https://pip.pypa.io) to install:
```shell
pip install pyceps
```
This will install all necessary dependencies and add a CLI entry point.
To test the installation run
```shell
pyceps --help
```

### Usage
To load a clinical mapping study, the system used must be specified and the location of the exported data must be given.
The study repository can either be a ZIP archive (preferred) or the top-level folder of the extracted data set.
```shell
pyceps.py --system "carto" --study-repository "path_to_repository"
pyceps.py --system "precision" --study-repository "path_to_repository"
```
This will gather basic information from the data set, i.e. name of the study, performed mapping procedures, etc.
The following commands are chained to this basic usage.

To import mapping procedures the name of the mapping procedure can be specified, or all available maps can be imported.
All information related to the mapping procedure is loaded, i.e. anatomical shell, mapping points, ablation lesions, etc.
```shell
pyceps.py ... --import-map "map_name"
pyceps.py ... --import-map "all"
```

A pickled version of the imported dataset can be saved:
```shell
pyceps.py ... --import-map "all" --save-study
```

To open a saved pickled version of the data set use
```shell
pyceps.py --system "carto" --pkl-file "path_to_pkl"
```
The system has to be specified again.
After loading additional data can be added to the data set if the study repository (original data) is still accessible.

Imported data sets can be visualized in a local HTML site to evaluate the quality of the data set:
```shell
pyceps.py ... --visualize
```

To export data in OPENcarp compatible formats (.pts, .elem, .igb, .dat) the following commands can be used
```shell
pyceps.py ... --dump-mesh
pyceps.py ... --dump-point-egms
pyceps.py ... --dump-point-ecgs
pyceps.py ... --dump-map-ecgs
pyceps.py ... --dump-surface-maps
pyceps.py ... --dump-lesions
```

### Advanced
The data contained in exported data sets differs for different mapping systems.
Exporting data via the CLI accesses only data common to every mapping system.
To access the entirety of imported data, Python scripts have to be used.

````python
from src import CartoStudy

study = CartoStudy("path_to_repository")
# import all available maps
study.import_maps(study.mapNames)
...
````

### License
This software is made available under the terms of the GNU Lesser General Public License v3.0

### How To Cite

### Whom Do I Talk To?
* [R. Arnold](mailto:robert.arnold@medunigraz.at?subject=pyceps)
* A. Prassl