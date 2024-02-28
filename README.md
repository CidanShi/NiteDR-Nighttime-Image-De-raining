# NiteDR-Nighttime-Image-De-raining
## NiteDR: Nighttime Image De-Raining with Cross-View Sensor Cooperative Learning for Dynamic Driving Scenes.

## Contents

- [Abstract](#Abstract)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Usage](#Usage)
- [NiteDR](#NiteDR)
  - [Illustration of our model](#Illustrationofourmodel)
  - [Qualitative Results](#QualitativeResults)
  - [Quantitative Results](#QuantitativeResults)
 
### Abstract
In real-world environments, outdoor imaging systems are often affected by disturbances such as rain degradation. Especially, in nighttime driving scenes, insufficient and uneven lighting shrouds the scenes in darkness, resulting degradation of both the image quality and visibility. Particularly, in the field of autonomous driving, the visual perception ability of RGB sensors experiences a sharp decline in such harsh scenarios. Additionally, driving assistance systems suffer from reduced capabilities in capturing and discerning the surrounding environment, posing a threat to driving safety. Single-view information captured by single-modal sensors cannot comprehensively depict the entire scene. To address these challenges, we developed an image de-raining framework tailored for rainy nighttime driving scenes. It aims to remove rain artifacts, enrich scene representation, and restore useful information. Specifically, we introduce cooperative learning between visible and infrared images captured by different sensors. By cross-view fusion of these multi-source data, the scene within the images gains richer texture details and enhanced contrast. We constructed an information cleaning module called CleanNet as the first stage of our framework. Moreover, we designed an information fusion module called FusionNet as the second stage to fuse the clean visible images with infrared images. Using this stage-by-stage learning strategy, we obtain de-rained fusion images with higher quality and better visual perception. Extensive experiments demonstrate the effectiveness of our proposed Cross-View Cooperative Learning (CVCL) in adverse driving scenarios in low-light rainy environments. The proposed approach addresses the gap in the utilization of existing rain removal algorithms in specific low-light conditions.
