# Demonstrator: Exploring Massive Sensor Data with Change Point Correlation

This repository contains an interactive demonstrator that allows to explore massive collections of time series
sensor data using Change Point Correlation. See the details on the prototype in our publication. The demonstrator
is currently hosted in this [live demo](https://changepoint.cs.fau.de).

See this [video](https://www.fau.tv/clip/id/63306) for short introduction and some of the main interactions.

If you are interested in the underlying methods, see our recent [publications](https://scholar.google.com/citations?hl=de&user=tItk-nIAAAAJ&view_op=list_works&sortby=pubdate)
or check out our pip-installable [package](https://github.com/Lucew/changepoynt) on algorithms for Change Point Detection.

## Short Explanation
Change Point Correlation (CPC) computes a similarity between time series that corresponds to simultaneous events. Instead of
comparing the raw time series, it uses methods from Change Point Detection to compute Change Scores and then compares 
these scores. Very generally said: What changes together, belongs together.

We use CPC and t-SNE to create a scatter plot, where each point corresponds to a signal and points that are close
together change a similar times.

See our publication for further details.

## Dataset
The prototype was initially designed for large scale sensor data analysis for complex power plants. Unfortunately, this
data comes with high confidentiality. To publicly present our prototype, we use a comparable
but public [dataset](https://doi.org/10.5281/zenodo.3563389) that was recorded from three different fishing boats. We
are very thankful to the members of the [European DataBio Project](https://doi.org/10.1007/978-3-030-71069-9) for
publishing the dataset under a permissive license.

## Running the Prototype
The prototype comes in the form of an interactive web application. We already pre-processed the sensor data and computed
the CPC similarity matrix. `Dash_Mainpage.py` is the main entrypoint into the application. 

Before running the project:

1) Clone the project using git-lfs to get access to all necessary files.

2) Install the necessary dependencies (we recommend using a virtual Python environment) `pip install -r requirements.txt` or `uv pip install -re requirements.txt`

Run the project with:
`python Dash_Mainpage.py`

We also provide a [container file](./Dockerfile) that can be used to run the application with podman or docker.

## Attributions

This repository includes processed data derived from:

**Siltanen, Pekka; Uriondo Arrue, Zigor.**
*Sensor data from three different fishing ships for a period of one month*.
Zenodo. DOI: [10.5281/zenodo.3563390](https://doi.org/10.5281/zenodo.3563390)

Original dataset license: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**

The version included here has been processed and modified for use with this application.
Please attribute the original authors and source when redistributing or reusing this data.