<div align="center">
<a href="https://wandb.ai">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" alt="Logo" width="150" height="150">
  </a>
  <a href="https://tensorflow.org">
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="Logo" width="150" height="150">
  </a>
<h3 align="center">Wandb + TensorFlow</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

It is a simple example demonstrates how to use the [Wandb](https://wandb.ai) API to monitor and analyze your Machine Learning projects.

With free Wandb account you will have a 100 GB of cloud storage for your logs, models, artifacts, and other data.

It also uses Docker and Docker Compose. And runs on GPU and CPU as well.

<!-- GETTING STARTED -->
## Getting Started

1. First thing first, you should have a [Wandb](https://wandb.ai) account to monitor your experiments.
2. You need a Wandb API key. You can get one from [Wandb account settings](https://wandb.ai/settings).


### Dependencies

**1. Python**
* tensorflow==2.9.1
* tensorflow_datasets==4.6.0
* PyYAML==6.0
* wandb==0.12.17 

**2. wandb account** for tracking your experiments.

* Create here: https://wandb.ai/signup

****



### Installation

1. Clone this repo
   ```sh
   git clone https://github.com/Alex-Kopylov/wandb-tensorflow-example.git
   ```
2. Setup your environment. You have several options:
   ```sh
   pip install -r requirements.txt
   ```

<a href="https://docs.docker.com/compose/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Docker_%28container_engine%29_logo.svg" alt="Docker Logo" height="100">
</a>

Docker is preferable for further integration in complex CI\/CD pipelines.
```sh
  docker compose build
  ```

<a href="https://docs.conda.io/en/latest/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ea/Conda_logo.svg" alt="Logo" height="50">
</a>

You can use Conda or default Python virtual environment.

```sh
  conda create -n wandb-tensorflow python=3.8
  conda activate wandb-tensorflow
  pip install -r requirements.txt
  ```