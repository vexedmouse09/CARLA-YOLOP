Unofficial 2024 CARLA YOLOP integration, 
currently the YOLOP direct integration is not functional, but the scripts I have created in order to make this process start to work are located in this repo. Future research is to be done in order to make this project work as expected. This is an implementation based directly off of SekiroRong's CARLA YOLOP integration and dataset generator code. 

Scripts pertaining to the possible integration:
- YOLP_2_CARLA.py
- automatic_control_with_yolop.py

Scripts pertaining to the FUNCTIONAL dataset generator:
-automatic_control_AND_MORE.py

To run the dataset generator, navigate to your working directory, and run the following command: 
```./automatic_control_AND_MORE.py``` you may also want to run a set of arguments to specify certain parameters, such as ```-n #``` for number of vehicles (# is any non-negative number, but I recommend you use a lower number to start to avoid system crashes. ```-w #``` specifies the amount of pedestrians, although this section of the code is currently commented out due to system crashes caused by pedestrian spawning. Other parameters need not be changed (unless absolutely neccesary), and many of the additional parameters may be changed from directly within the script itself (there are accompanying comments within the script.


In order to use this script , you may follow the directions provided by SekiroRong in this repo for basic instructions, as our code follows their framework with a few minor additions to ensure proper functionality: 
https://github.com/SekiroRong/Carla_dataset_generator

Throughout the scripts in this repo, our team has made many comments in order to help readers understand what each portion of the script does.

In order to properly set up the CARLA simulator, please follow the directions for building the simulator from source, as this will give you much more control over the environments the car will be driving in . 
I have included a copy of their README below, as well as copies of SekiroRong's READMEs within the folders. 

The README below originates from the following repository: https://github.com/carla-simulator/carla



CARLA Simulator Official README
===============

[![Documentation](https://readthedocs.org/projects/carla/badge/?version=latest)](http://carla.readthedocs.io)

[![carla.org](Docs/img/btn/web.png)](http://carla.org)
[![download](Docs/img/btn/download.png)](https://github.com/carla-simulator/carla/blob/master/Docs/download.md)
[![documentation](Docs/img/btn/docs.png)](http://carla.readthedocs.io)
[![forum](Docs/img/btn/forum.png)](https://github.com/carla-simulator/carla/discussions)
[![discord](Docs/img/btn/chat.png)](https://discord.gg/8kqACuC)

CARLA is an open-source simulator for autonomous driving research. CARLA has been developed from the ground up to support development, training, and
validation of autonomous driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings,
vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites and
environmental conditions.

[![CARLA Video](Docs/img/0_9_15_thumbnail.webp)](https://www.youtube.com/watch?v=q4V9GYjA1pE )

### Download CARLA

Linux:
* [**Get CARLA overnight build**](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/Dev/CARLA_Latest.tar.gz)
* [**Get AdditionalMaps overnight build**](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/Dev/AdditionalMaps_Latest.tar.gz)

Windows:
* [**Get CARLA overnight build**](https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/Dev/CARLA_Latest.zip)
* [**Get AdditionalMaps overnight build**](https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/Dev/AdditionalMaps_Latest.zip)

### Recommended system

* Intel i7 gen 9th - 11th / Intel i9 gen 9th - 11th / AMD ryzen 7 / AMD ryzen 9
* +32 GB RAM memory
* NVIDIA RTX 3070 / NVIDIA RTX 3080 / NVIDIA RTX 4090
* Ubuntu 20.04

## Documentation

The [CARLA documentation](https://carla.readthedocs.io/en/latest/) is hosted on ReadTheDocs. Please see the following key links:

- [Building on Linux](https://carla.readthedocs.io/en/latest/build_linux/)
- [Building on Windows](https://carla.readthedocs.io/en/latest/build_windows/)
- [First steps](https://carla.readthedocs.io/en/latest/tuto_first_steps/)
- [CARLA asset catalogue](https://carla.readthedocs.io/en/latest/catalogue/)
- [Python API reference](https://carla.readthedocs.io/en/latest/python_api/)
- [Blueprint library](https://carla.readthedocs.io/en/latest/bp_library/)

## CARLA Ecosystem
Repositories associated with the CARLA simulation platform:

* [**CARLA Autonomous Driving leaderboard**](https://leaderboard.carla.org/): Automatic platform to validate Autonomous Driving stacks
* [**Scenario_Runner**](https://github.com/carla-simulator/scenario_runner): Engine to execute traffic scenarios in CARLA 0.9.X
* [**ROS-bridge**](https://github.com/carla-simulator/ros-bridge): Interface to connect CARLA 0.9.X to ROS
* [**Driving-benchmarks**](https://github.com/carla-simulator/driving-benchmarks): Benchmark tools for Autonomous Driving tasks
* [**Conditional Imitation-Learning**](https://github.com/felipecode/coiltraine): Training and testing Conditional Imitation Learning models in CARLA
* [**AutoWare AV stack**](https://github.com/carla-simulator/carla-autoware): Bridge to connect AutoWare AV stack to CARLA
* [**Reinforcement-Learning**](https://github.com/carla-simulator/reinforcement-learning): Code for running Conditional Reinforcement Learning models in CARLA
* [**RoadRunner**](https://www.mathworks.com/products/roadrunner.html): MATLAB GUI based application to create road networks in the ASAM OpenDRIVE format
* [**Map Editor**](https://github.com/carla-simulator/carla-map-editor): Standalone GUI application to enhance RoadRunner maps with traffic lights and traffic signs information


**Like what you see? Star us on GitHub to support the project!**

Paper
-----

If you use CARLA, please cite our CoRL’17 paper.

_CARLA: An Open Urban Driving Simulator_<br>Alexey Dosovitskiy, German Ros,
Felipe Codevilla, Antonio Lopez, Vladlen Koltun; PMLR 78:1-16
[[PDF](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)]
[[talk](https://www.youtube.com/watch?v=xfyK03MEZ9Q&feature=youtu.be&t=2h44m30s)]


```
@inproceedings{Dosovitskiy17,
  title = {{CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
```

Building CARLA
--------------

Clone this repository locally from GitHub:

```sh
git clone https://github.com/carla-simulator/carla.git .
```

Also, clone the [CARLA fork of the Unreal Engine](https://github.com/CarlaUnreal/UnrealEngine) into an appropriate location:

```sh
git clone --depth 1 -b carla https://github.com/CarlaUnreal/UnrealEngine.git .
```

Once you have cloned the repositories, follow the instructions for [building in Linux][buildlinuxlink] or [building in Windows][buildwindowslink].

[buildlinuxlink]: https://carla.readthedocs.io/en/latest/build_linux/
[buildwindowslink]: https://carla.readthedocs.io/en/latest/build_windows/

Contributing
------------

Please take a look at our [Contribution guidelines][contriblink].

[contriblink]: https://carla.readthedocs.io/en/latest/cont_contribution_guidelines/

F.A.Q.
------

If you run into problems, check our
[FAQ](https://carla.readthedocs.io/en/latest/build_faq/).

Licenses
-------

#### CARLA licenses

CARLA specific code is distributed under MIT License.

CARLA specific assets are distributed under CC-BY License.

#### CARLA Dependency and Integration licenses

The ad-rss-lib library compiled and linked by the [RSS Integration build variant](Docs/adv_rss.md) introduces [LGPL-2.1-only License](https://opensource.org/licenses/LGPL-2.1).

Unreal Engine 4 follows its [own license terms](https://www.unrealengine.com/en-US/faq).

CARLA uses three dependencies as part of the SUMO integration:
- [PROJ](https://proj.org/), a generic coordinate transformation software which uses the [X/MIT open source license](https://proj.org/about.html#license).
- [SQLite](https://www.sqlite.org), part of the PROJ dependencies, which is [in the public domain](https://www.sqlite.org/purchase/license).
- [Xerces-C](https://xerces.apache.org/xerces-c/), a validating XML parser, which is made available under the [Apache Software License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

CARLA uses one dependency as part of the Chrono integration:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), a C++ template library for linear algebra which uses the [MPL2 license](https://www.mozilla.org/en-US/MPL/2.0/).

CARLA uses the Autodesk FBX SDK for converting FBX to OBJ in the import process of maps. This step is optional, and the SDK is located [here](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0)

This software contains Autodesk® FBX® code developed by Autodesk, Inc. Copyright 2020 Autodesk, Inc. All rights, reserved. Such code is provided "as is" and Autodesk, Inc. disclaims any and all warranties, whether express or implied, including without limitation the implied warranties of merchantability, fitness for a particular purpose or non-infringement of third party rights. In no event shall Autodesk, Inc. be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of such code."
