# Galaxy Builder data aggregation

This repository contains a suite of software utilities which allow the aggregation, rendering and fitting of photometric models created by citizen science volunteers as part of the [Galaxy Builder](https://www.zooniverse.org/projects/tingard/galaxy-builder/) Zooniverse project.

## Getting Started

This package allows a number of different manipulations of Galaxy Builder models, including aggregating many models using `gzbuilder_analysis.aggregation`, calculating aggregate logarithmic spirals using `gzbuilder_analysis.aggregation.spirals`, rendering volunteer and aggregate models using `gzbuilder_analysis.rendering` and tuning models using `gzbuilder_analysis.fitting`.


## Examples 

A series of examples can be found in the `examples` folder, including 

- [A full run through of rendering a synthetic image, aggregating volunteer classifications and fitting the aggregate model](examples/full-worked-example.ipynb)
- [An in-depth look at the Jaccard clustering method](examples/jaccard-clustering.ipynb)
- [An in-depth look at Spiral arm clustering](examples/spiral-clustering.ipynb)
- An in-depth look at model fitting (TBA)

## Contributing

If you would like to help, pull requests are always welcome! If you would like access to a Galaxy Builder data dump, please contact the repository owner(s).
