<h1 align = "center">Flask Image Processing API</h1>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#about-the-project">About The Project</a> </li>
    <li> <a href="#built-with">Built With</a> </li>
    <li> <a href="#installation">Installation</a> </li>
    <li> <a href="#usage">Usage</a> </li>
    <li> <a href="#model-details">Model Details</a> </li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About the Project

The aim of this project is to implement a flask API. The primary function of the API is to accept an image, perform image classification using a pre-trained model, and return the results in the form of a JSON object.

The project was built for an assignment, and is meant to be a learning-based project. It is not meant to be a tool for direct pratical use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With
* [![Python][Python-badge]][Python-url]
* [![Flask][Flask-badge]][Flask-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Installation -->
## Installation

The following dependencies need to be installed before being able to run the project:
<ul>
  <li>flask (version 2.25 or greater)</li>
  <li>numpy (version 1.23.5 or greater)</li>
  <li>tensorflow (version 2.12.0 or greater)</li>
  <li>tensorflow hub (version 0.8.0 or greater)</li>
</ul>

It is preferred to use a package manager such as anaconda to perform the installation of dependencies as well as the execution of the API, in order to prevent conflicts with other pre-existing libraries.

After installing conda, execute the following to setup the environment using the environment.yml file:
```
conda env create -f environment.yml
```

The environment can then be activated as follows before use:
```
conda activate flask-API
```

The project has been tested on a system running Ubuntu 22.04.3 LTS, and is guarenteed to work reliably on the same without any alterations required to the list of dependencies. In order to run the API on another system, kindly resolve any dependency conflicts (version conflicts, missing functionality, etc.) before use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Usage -->
## Usage

The API consists of a single python file, and can be used by first starting the API server, then sending requests to it through curl or any other request framework.

To start the server, navigate to the directory containing the project directory, activate the corresponding virtual environment, and run the python script as follows:

```
python3 -m API.py
```

After the server has been started, requests can be sent to it through a separate terminal instance. The image to be sent to the API must be present on the system through which the request is being sent. All common image formats are accepted. The general syntax for the same is as follows (using curl version 7.81.0):
```
curl -X POST http://<localhost IP>:<port>/upload -F "image=@<path to image to be uploaded>"
```

Example (the image uploaded has been tested, with the results shown below):
```
curl -X POST http://127.0.0.1:5000/upload -F "image=@./football.bmp"
```


The expected result is a JSON object containing the top three predictions made by the pre-trained model, along with the confidence level for each prediction.

Example (formatted for clarity):
```
{
  "prediction_1":
    {
      "confidence_level":0.41517,
      "predicted_label":"soccer ball"
    },
  "prediction_2":
    {
      "confidence_level":0.21616,
      "predicted_label":"electric fan"
    },
  "prediction_3":
    {
      "confidence_level":0.11925,
      "predicted_label":"chainlink fence"
    }
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Model Details -->
## Model Details
The pre-trained model use is Google's mobile-net v2.

This is a SSD-based object detection model trained on Open Images V4 with ImageNet pre-trained MobileNet V2 as image feature extractor.

It can be found on kaggle at: https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[Flask-badge]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/3.0.x/