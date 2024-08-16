cd ..

source venv/bin/activate

for i in $(seq 17 24); do
    python train_inr.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --neutral_path=data/tetmesh_face_surface.obj \
        --deformed_path="data/ground_truths/deformed_surface_0${i}.obj" \
        --checkpoint_path="checkpoints/best_model_0${i}_pair_healthy.pth" \
        --use_pretrained \
        --pretrained_path=checkpoints/prior.pth \
        --predicted_jaw_path=data/predicted_jaw.npy \
        --train
    
    python flip_surface.py --neutral_surface_path=data/tetmesh_face_surface.obj \
        --deformed_surface_path="data/ground_truths/deformed_surface_0${i}.obj" \
        --contour_path=data/tetmesh_contour.obj \
        --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
        --deformed_out_path="data/deformed_out_0${i}.obj"

    python train_inr.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --neutral_path=data/tetmesh_face_surface.obj \
        --deformed_path="data/deformed_out_0${i}.obj" \
        --checkpoint_path="checkpoints/best_model_0${i}_pair_unhealthy.pth" \
        --use_pretrained \
        --pretrained_path=checkpoints/prior.pth \
        --predicted_jaw_path=data/predicted_jaw.npy \
        --train

    PYTHONPATH=. python actuation_simulator.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --predicted_jaw_path=data/predicted_jaw.npy \
        --main_actuation_model_path="checkpoints/best_model_0${i}_pair_healthy.pth" \
        --secondary_actuation_model_path="checkpoints/best_model_0${i}_pair_unhealthy.pth" \
        --contour_path=data/tetmesh_contour.obj \
        --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
        --use_pretrained \
        --pretrained_path=checkpoints/prior.pth \
        --checkpoint_path="checkpoints/best_model_simulator_0${i}_pair.pth" \
        --train
done
