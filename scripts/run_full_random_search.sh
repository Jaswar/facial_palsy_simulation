cd ..

source venv/bin/activate

for i in $(seq 1 1000); do
    python random_search.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --neutral_path=data/tetmesh_face_surface.obj \
        --deformed_path=data/ground_truths/deformed_surface_003.obj \
        --model_path=checkpoints/random_search/best_model_rs_$i.pth \
        --config_path=checkpoints/random_search/best_config_rs_$i.json \
        --predicted_jaw_path=data/random_search/predicted_jaw_003_$i.npy \
        --budget=20 \
        --num_runs=100

    python random_search.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --main_actuation_model_path=checkpoints/best_model_003_pair_healthy_5_256.pth \
        --secondary_actuation_model_path=checkpoints/best_model_003_pair_unhealthy_5_256.pth \
        --contour_path=data/tetmesh_contour.obj \
        --reflected_contour_path=data/tetmesh_contour_ref_deformed.obj \
        --model_path=checkpoints/random_search/best_model_rs_sim_$i.pth \
        --config_path=checkpoints/random_search/best_config_rs_sim_$i.json \
        --use_pretrained \
        --pretrained_model_path=checkpoints/prior_8_5_256.pth \
        --predicted_jaw_path=data/predicted_jaw_003_$i.npy \
        --num_runs=-1 \
        --budget=20 \
        --simulator
done

