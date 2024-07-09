# ActiMotus
- [Citation](#citing-actimotus)
- [How it works](#how-actimotus-works)
- [Versions](#versions)
- [Get in touch](#get-in-touch)
- [Contributions and license](#contributions-and-license)

ActiMotus is the data processing algorithms of the Motus system and are developed based on the [Acti4 software](https://github.com/motus-nfa/Acti4 "Acti4 GitHub repository").\
Motus is an activity detection system developed by the National Research Centre for the Working Environment ([NFA](https://nfa.dk "NFA's homepage")).\
**You can read more about the entire Motus system on our [webpage](http://motus-system.notion.site "Motus webpage").**

In this repository you can find the Motus source code.

## Citing ActiMotus
If you use Motus in your research please cite the system as follows:

APA
```
The National Research Centre for the Working Environment (NFA). (2024). ActiMotus (2.0.0) [Computer software]. https://github.com/motus-nfa/Motus
```

BibTex
```
@software{The_National_Research_Centre_for_the_Working_Environment_NFA_ActiMotus_2024,
author = {The National Research Centre for the Working Environment (NFA)},
month = jul,
title = {{ActiMotus}},
url = {https://github.com/motus-nfa/Motus},
version = {2.0.0},
year = {2024}
}
```

## How ActiMotus works
ActiMotus is based on the physical activity classification software, [Acti4](https://github.com/motus-nfa/Acti4 "Acti4 GitHub repository"). While Acti4 was written in MatLab and developed as an offline software, ActiMotus is written in Python and runs on a cloud service hosted by [SENSmotion](https://www.sens.dk/da/ "SENS website"). ActiMotus contains the same core algorithms and concepts as Acti4, but differs in the following ways (due to back-end memory limitiations):

### The execution flow of the scripts is split into two main parts, preprocessing and activity classification
ActiMotus processes accelerometer data in two steps. <br>
Step 1 reads the raw data in 12 hour chunks and runs pre-processing steps. These steps include 
- Detecting sampling frequency (either 25 Hz or 12.5 Hz).
- Detecting periods where accelerometer is not worn.
- Computing rolling means and standard deviations of the three axes for overlapping windows of 2 seconds (overlap of 1 second), which are the primary features of the activity classification. 
- Computing metrics used in the detection of cycling, estimation of the step count and in rotating the data based on a reference angle. These metrics are used in place of the raw data, which is unavailable in the second step. 

The outputs from the processing of step 1 are used as inputs to step 2. In step 2 multiple chunks are read simultaneously such that data is processed for 24 hours at a time. 
In the second step, the sensor placement is revealed and, depending on this, different processes are activated. From a thigh worn accelerometer, the primary physical activities are classified using the Acti4 algorithm. Further sensors both improve the primary classification and produces angle-inclination signals. The output of step 2 is thus the classified activities on a second to second basis, along with step counts and angle signals (also on a second to second basis). 


### Sensor placements

- **Thigh** enables the primary activity classification.
- **Calf** sensor adds the ability to detect kneeling and squatting.
- **Upper back** enchances the detection of lying compared to thigh-only measures. Further, forward bending of the back is measured.
- **Arm** is used to capture arm inclinations and movement.

New algorithms and sensors are added continuously.

## Versions
ActiMotus is constantly developed and improved.  
The current stable and integrated version of ActiMotus is 2.0.0  
For an overview of versions and compatibilities please review [our version overwiev](./doc/Versions.txt).

## Get in touch
If you want to use Motus or are interested in knowing more, please contact motus@nfa.dk with your request and contact information.  

## Contributions and license
ActiMotus is published under the [BSD 3-Clause License](./LICENSE).  
All contributions to Motus are welcome. If you have suggestions to features, you can open an issue or create a pull request with suggested changes.
