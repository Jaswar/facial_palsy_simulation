for i in $(seq 1 10000); do
    source venv/bin/activate

    python random_search.py --model_path=checkpoints/random_search/model_$i.pth --config_path=checkpoints/random_search/config_$i.json
    python predict_actuations.py --model_path=checkpoints/random_search/model_$i.pth --actuations_path=data/random_search/actuations_$i.npy --config_path=checkpoints/random_search/config_$i.json
    python tetmesh_symmetry_hack.py --act_fn=random_search/actuations_$i.npy --act_out_fn=random_search/act_sym_$i.npy

    deactivate

    cd ../FP-FEM
    source venv/bin/activate

    PYTHONPATH=. python face_simulation/2_face_sim_2_forward.py --fn_act_sym=../ssrf_project/data/random_search/act_sym_$i.npy --deformed_nodes_path=deformed_nodes_$i.npy

    deactivate
    cd ../ssrf_project
done