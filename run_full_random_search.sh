source venv/bin/activate

for i in $(seq 1 1000); do
    python random_search.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --neutral_path=data/tetmesh_face_surface.obj \
        --deformed_path=data/ground_truths/deformed_surface_001.obj \
        --model_path=checkpoints/random_search/best_model_rs_$i.pth \
        --config_path=checkpoints/random_search/best_config_rs_$i.json \
        --predicted_jaw_path=data/random_search/predicted_jaw_001_$i.npy \
        --budget=20 \
        --num_runs=1
    
    python predict_actuations.py --tetmesh_path=data/tetmesh \
        --tetmesh_contour_path=data/tetmesh_contour.obj \
        --tetmesh_reflected_deformed_path=data/tetmesh_contour_ref_deformed.obj \
        --model_path=checkpoints/random_search/best_model_rs_$i.pth \
        --config_path=checkpoints/random_search/best_config_rs_$i.json \
        --out_actuations_path=data/random_search/act_sym_001_$i.npy \
        --silent

    python random_search.py --tetmesh_path=data/tetmesh \
        --jaw_path=data/jaw.obj \
        --skull_path=data/skull.obj \
        --neutral_path=data/tetmesh_face_surface.obj \
        --deformed_path=data/ground_truths/deformed_surface_001.obj \
        --model_path=checkpoints/random_search/best_model_rs_sim_$i.pth \
        --config_path=checkpoints/random_search/best_config_rs_sim_$i.json \
        --predicted_jaw_path=data/random_search/predicted_jaw_001_$i.npy \
        --num_runs=1 \
        --budget=20 \
        --simulator \
        --actuations_path=data/random_search/act_sym_001_$i.npy
done

