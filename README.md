# ETH SSRF Project

### Main method

The main method consists of multiple stages: training the default INR, training the flipped INR, and simulating the combination of the two. Each of these stages is done with a separate script.

The first stage is done with the `train_inr.py` script. Below is an example on how to run the script:

```bash
python train_inr.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_017.obj \
    --checkpoint_path=checkpoints/best_model.pth \
    --use_pretrained \
    --pretrained_path=checkpoints/prior.pth \
    --predicted_jaw_path=data/predicted_jaw.npy \
    --train
```

Next, one can generate the second INR for the flipped deformed surface. To do so, one needs to run the script `flip_surface.py` followed by another
call to `train_inr.py` with `deformed_path` being set to this new flipped surface. Running `flip_surface.py` can be done with the following:

```bash
python flip_surface.py --neutral_surface_path=data/tetmesh_face_surface.obj \
    --deformed_surface_path=data/ground_truths/deformed_surface_017.obj \
    --contour_path=data/tetmesh_contour.obj \
    --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
    --deformed_out_path=data/deformed_out.obj
```

Next, to simulate the generated actuations, the following script must be run:

```bash
PYTHONPATH=. python actuation_simulator.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --predicted_jaw_path=data/predicted_jaw.npy \
    --main_actuation_model_path=checkpoints/best_model_017_pair_healthy.pth \
    --secondary_actuation_model_path=checkpoints/best_model_017_pair_unhealthy.pth \
    --contour_path=data/tetmesh_contour.obj \
    --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
    --use_pretrained \
    --pretrained_path=checkpoints/prior.pth \
    --checkpoint_path=checkpoints/best_model_simulator.pth \
    --train
```

Finally, one can visualize the effect of applying the learned simulation model to the high resolution mesh. This can be done 
with the `predict_high_res.py` file in the following way:

```bash
python predict_high_res.py --neutral_path=data/tetmesh_face_surface.obj \
    --high_res_path=../medusa_scans/rawMeshes/take_001.obj \
    --model_path=checkpoints/best_model_simulator_017_pair.pth
```

INR for the surface:
```bash
python train_surface_inr.py --neutral_path=data/tetmesh_face_surface.obj \
    --neutral_flame_path=../flame-fitting/output/fit_scan_result_neutral_surface_no_landmarks.obj \
    --deformed_flame_path=../flame-fitting/output/fit_scan_result_001_symmetric_loss_scan.obj \
    --checkpoint_path=checkpoints/best_model_surface_inr_001.pth \
    --train
```

Creating figures:
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
    --save_path=screenshots/figure_001.png \
    --flame_path=../flame-fitting/output/fit_scan_result_001_neutral_midpoint.obj
```

### Random search

`random_search.py` performs a random search over specified hyperparameters. The following command will launch it:

```bash
python random_search.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_017.obj \
    --model_path=checkpoints/best_model_rs.pth \
    --config_path=checkpoints/best_config_rs.json
```

The script `run_full_random_search.sh` runs the random search for both the INR and the simulator. The results can be visualized qualitatively with the `visualize_full_random_search.py` script, which can be run with:

```bash
python visualize_full_random_search.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_017.obj \
    --checkpoints_path=checkpoints/random_search
```

### UV baseline

The UV baseline is built in the `uv_baseline.py` script. To run the baseline, a face surface with the UV mesh must be extracted (can be done with Blender). Then, the script can be run with:

```bash
python uv_baseline.py --path=data/face_surface_with_uv3.obj \
    --deformed_path=data/ground_truths/deformed_surface_023.obj \
    --laplace
```

