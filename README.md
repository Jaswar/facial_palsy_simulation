# Predicting Healthy Facial Expressions in Patients with Facial Palsy

Implementation for the paper "Predicting Healthy Facial Expressions in Patients with Facial Palsy". 

Please note that this paper is accompanied by the (slightly modified) FLAME fitting repository, available [here](https://github.com/Jaswar/flame-fitting). The original repository can be found [here](https://github.com/Rubikplayer/flame-fitting).

The data has not been made available publicly. The pretrained models are not made public as they can easily be used to generate the original data.

## Installation

The required dependencies are provided in the `requirements.txt` file. It is best to install those in a new Python 3.10 virtual environment.

## Running the models

### FLAME fitting model

First, a FLAME model needs to be fitted. This needs to be done using the secondary repository (see above).

The as-rigid-as-possible network can then be trained with:

```bash
python train_surface_inr.py --neutral_path=data/tetmesh_face_surface.obj \
    --neutral_flame_path=../flame-fitting/output/fit_scan_result_neutral_surface_no_landmarks.obj \
    --deformed_flame_path=../flame-fitting/output/fit_scan_result_001_neutral_midpoint.obj \
    --checkpoint_path=checkpoints/best_model_surface_inr_001.pth \
    --train
```

To make the prediction on the high-resolution surface, the following should be run:

```bash
python predict_high_res.py --neutral_path=data/tetmesh_face_surface.obj \
    --high_res_path=../medusa_scans/rawMeshes/take_001.obj \
    --model_path=checkpoints/best_model_surface_inr_001.pth
```

### Muscle actuation model

First, the network for the healthy side of the face needs to be trained. This can be done as follows:

```bash
python train_inr.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_001.obj \
    --checkpoint_path=checkpoints/best_model_001_pair_healthy.pth \
    --predicted_jaw_path=data/predicted_jaw.npy \
    --train
```

Next, the network for the unhealthy side has to be trained. For this, the surface first needs to be flipped as below. 

```bash
python flip_surface.py --neutral_surface_path=data/tetmesh_face_surface.obj \
    --deformed_surface_path=data/ground_truths/deformed_surface_017.obj \
    --contour_path=data/tetmesh_contour.obj \
    --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
    --deformed_out_path=data/deformed_out.obj
```

The second network can then be trained as follows.

```bash
python train_inr.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_001.obj \
    --checkpoint_path=checkpoints/best_model_001_pair_unhealthy.pth \
    --train
```

Next, to simulate the generated actuations, the following script must be run:

```bash
PYTHONPATH=. python actuation_simulator.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --predicted_jaw_path=data/predicted_jaw.npy \
    --main_actuation_model_path=checkpoints/best_model_001_pair_healthy.pth \
    --secondary_actuation_model_path=checkpoints/best_model_001_pair_unhealthy.pth \
    --contour_path=data/tetmesh_contour.obj \
    --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
    --checkpoint_path=checkpoints/best_model_simulator_001_pair.pth \
    --train
```

The prediction on the high-res surface can then be done as follows:

```bash
python predict_high_res.py --neutral_path=data/tetmesh_face_surface.obj \
    --high_res_path=../medusa_scans/rawMeshes/take_001.obj \
    --model_path=checkpoints/best_model_simulator_001_pair.pth
```

### Random search

The random search can be run with the `random_search.py` script and visualized using the `visualize_full_random_search.py` script. For usage examples, check the `scripts` folder.

Visualization should only be used when all parameters are optimized, such that the results can be assessed qualitatively.

### Creating figures

Images for the figures as presented in the paper can be generated using the `create_figures.py` script as shown below. Refer to the script for more usages. 

```bash
python create_figures.py --neutral_path=data/tetmesh_face_surface.obj \
    --high_res_path=../medusa_scans/rawMeshes_ply/take_001.ply \
    --simulator_model_path=checkpoints/best_model_simulator_001_pair.pth \
    --healthy_inr_model_path=checkpoints/best_model_001_pair_healthy.pth \
    --unhealthy_inr_model_path=checkpoints/best_model_001_pair_unhealthy.pth \
    --original_high_res_path=../medusa_scans/rawMeshes_ply/take_002.ply \
    --original_low_res_path=data/ground_truths/deformed_surface_001.obj \
    --tetmesh_path=data/tetmesh \
    --contour_path=data/tetmesh_contour.obj \
    --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
    --flame_path=../flame-fitting/output/fit_scan_result_001_neutral_midpoint.obj \
    --save_path=screenshots/figure_001.png
```