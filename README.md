# ETH SSRF Project

### Main method

The main method consists of three stages: training an INR, generating symmetric actuations, simulating the generated actuations with a neural simulator. Each of these stages is done with a separate script.

The first stage is done with the `train_inr.py` script. Below is an example on how to run the script:

```bash
python train_inr.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_017.obj \
    --checkpoint_path=checkpoints/best_model.pth \
    --train
```

Next, the actuations have to be generated and symmetrised. This is done with the `predict_actuations.py` script:

```bash
python predict_actuations.py --tetmesh_path=data/tetmesh \
    --tetmesh_contour_path=data/tetmesh_contour.obj \
    --tetmesh_reflected_deformed_path=data/tetmesh_contour_ref_deformed.obj \
    --model_path=checkpoints/best_model_017.pth \
    --config_path=checkpoints/best_config.json \
    --out_actuations_path=data/act_sym_017_per_vertex.npy
```

Finally, to simulate the generated actuations, the following script must be run:

```bash
python actuation_simulator.py --tetmesh_path=data/tetmesh \
    --jaw_path=data/jaw.obj \
    --skull_path=data/skull.obj \
    --neutral_path=data/tetmesh_face_surface.obj \
    --deformed_path=data/ground_truths/deformed_surface_017.obj \
    --actuations_path=data/act_sym_017_per_vertex.npy \
    --checkpoint_path=checkpoints/best_model_simulator.pth \
    --train
```

### UV baseline

The UV baseline is built in the `uv_baseline.py` script. To run the baseline, a face surface with the UV mesh must be extracted (can be done with Blender). Then, the script can be run with:

```bash
python uv_baseline.py --path=data/face_surface_with_uv3.obj \
    --deformed_path=data/ground_truths/deformed_surface_023.obj \
    --laplace
```

