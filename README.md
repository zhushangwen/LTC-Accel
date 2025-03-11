# Accelerating Diffusion Sampling via Exploiting Local Transition Coherence

This project aims to introduce a training-free method, LTC-Accel, which accelerates the sampling process of diffusion models by identifying **Local Transition Coherence** and implementing corresponding acceleration strategies. Here we just present an example on EDM based on Stable Diffusion v3.5, and you can feel free to change the base model and scheduler since LTC-Accel is training-free and widely compatiable. The project includes two main components: the original model and scheduler (`main.py`) and the accelerated sampling strategy (`step.py`).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Original Model](#running-the-original-model)
  - [Running the Accelerated Sampling](#running-the-accelerated-sampling)
- [Visualization](#visualization)
- [Other Implementations and Plugins](#other-implementations-and-plugins)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diffusion models have shown remarkable performance in text-based generation, but their sampling process can be computationally intensive. This project explores the phenomenon of **Local Transition Coherence** in the sampling process and implements strategies to accelerate the sampling process without compromising the quality of the generated samples.

The project consists of two main files:

- `ddimx.py`: Implements the original diffusion model and scheduler.
- `step.py`: Implements the accelerated sampling strategy.

## Installation

To use this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/zhushangwen/LTC-Accel.git
cd ./LTC-Accel
```

## Usage
### Running the Original Model
To run the original diffusion model and scheduler, use the `ddimx.py` script and set parameters as the followings:

```python
images = pipe(prompt, device=device, num_inference_steps = inference_steps, cal_wg = False, skip_x = False).images
```

You can feel free to change the parameters as long as `skip_x = False`. Then you can run the original model:
```bash
python main.py
```

### Running the Accelerated Sampling
To run the accelerated sampling process, first it is necessary to obtain one important parameter $w_g$ used for measuring the **Local Transition Coherence** and approximating some sampling steps. Specifically, set `cal_wg = True` and `skip_x = False` as the following example:

```python
images = pipe(prompt, device=device, num_inference_steps = inference_steps, cal_wg = True, skip_x = False).images
```

Then, use the $w_g$ to run the accelerated sampling of the original sampling process:

```python
images = pipe(prompt, device=device, num_inference_steps = inference_steps, cal_wg = False, skip_x = True).images
```
```bash
python main.py
```

You can feel free to modify the parameters concerning the acceleration condition in `step.py` as you want. The following is one example for `mod` and `skip_cond`:
```python
mod = 2
skip_cond = (i % mod == mod - 1 and i > 20)
```

## Visualization
The accelerated sampling strategy implemented in this project has been tested on various datasets and models. Below are some images generated through the original and accelerated sampling:
<div style="position: relative; display: inline-block;">
  <img src="example.png" alt="Sampling Time Comparison" width="800">
  <!-- 根据实际图片位置调整 top 和 left 的值 -->
  <div style="position: absolute; top: 20px; left: 60px; background-color: rgba(255,255,255,0.5); padding: 2px 4px;">
    <strong>8-Step LTC-Accel</strong>
  </div>
  <div style="position: absolute; top: 20px; left: 210px; background-color: rgba(255,255,255,0.5); padding: 2px 4px;">
    <strong>8-Step Original</strong>
  </div>
  <div style="position: absolute; top: 20px; left: 360px; background-color: rgba(255,255,255,0.5); padding: 2px 4px;">
    <strong>12-Step Original</strong>
  </div>
</div>

Specifically, the first column presents images generated from 8-step LTC-Accel accelerated from the 12-step original sampling process, while the second and the third column are generated from 8-step and 12-step original sampling process.

## Other Implementations and Plugins
We sincerely thank the authors listed below who implemented LTC-Accel in plugins or other contexts.

- Diffusers: https://huggingface.co/docs/diffusers

## Contributing
We welcome contributions to this project! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request.

- Fork the repository.
- Create a new branch (`git checkout -b feature/YourFeatureName`).
- Commit your changes (`git commit -m 'Add some feature'`).
- Push to the branch (`git push origin feature/YourFeatureName`).
- Open a pull request.

## License
This project is licensed under the TODO - see the LICENSE file for details.
