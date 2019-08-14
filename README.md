# Galaxy Builder data aggregation

This repository contains a suite of software utilities which allow the aggregation, rendering and fitting of photometric models created by citizen science volunteers as part of the [Galaxy Builder](https://www.zooniverse.org/projects/tingard/galaxy-builder/) Zooniverse project.

## Getting Started

This package allows a number of different manipulations of Galaxy Builder models, including aggregating many models using `gzbuilder_analysis.aggregation`, calculating aggregate logarithmic spirals using `gzbuilder_analysis.spirals`, rendering volunteer and aggregate models using `gzbuilder_analysis.rendering` and tuning models using `gzbuilder_analysis.fitting`.

A fairly comprehensive overview of what you can do can be seen in [this PDF](https://github.com/tingard/gzbuilder_analysis/tree/master/example_use.pdf). It is worth noting that some custom code used there is not publicly available (yet), namely the `galaxy_utilities.py` script used to obtain classifications and galaxy metadata from pre-calculated files.


## Contributing

If you would like to help, pull requests are always welcome! If you would like access to a Galaxy Builder data dump, please contact the repository owners.
