"""
author: daniel dorda
"""

from pathlib import Path
import numpy as np
import pyvista as pv


SAVE_PTH_CLIP = Path("./clipped_mesh")


class Tetmesh:

    def __init__(self, fn_base, compute_face_to_ele_map=False):
        self.fn_base = Path(fn_base)
        self.nodes, self.eles, self.faces = self.read_tetgen_file(self.fn_base)
        if compute_face_to_ele_map:
            self.face_to_ele = self.map_face_to_ele(self.faces, self.eles)
        else:
            self.face_to_ele = None

    @staticmethod
    def read_tetgen_file(fn_base, require_face_file=False):

        node_file = Path(str(fn_base) + '.node')
        ele_file = Path(str(fn_base) + '.ele')
        face_file = Path(str(fn_base) + '.face')

        assert node_file.exists() and ele_file.exists()
        if require_face_file:
            assert face_file.exists()

        with open(node_file, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split() for l in lines]  # turn into lists of symbols
            if lines[0][0] == "#":
                lines = lines[1:]

            node_num = int(lines[0][0])
            # ignore attributes for now
            nodes = np.asarray([[float(v) for v in lines[i + 1][1:4]] for i in range(node_num)])

        with open(ele_file, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split() for l in lines]
            if lines[0][0] == "#":
                lines = lines[1:]

            ele_num = int(lines[0][0])
            elements = np.asarray([[int(e) for e in lines[i + 1][1:5]] for i in range(ele_num)], dtype=int)
            if np.min(elements) == 1:
                elements = elements - 1

        faces = None
        if face_file.exists():
            with open(face_file, 'r') as f:
                lines = f.readlines()
                lines = [l.strip().split() for l in lines]
                face_num = int(lines[0][0])
                faces = np.asarray([[int(e) for e in lines[i + 1][1:4]] for i in range(face_num)], dtype=int)
                if np.min(faces) == 1:
                    faces = faces - 1

        return nodes, elements, faces

    @staticmethod
    def map_face_to_ele(faces, eles):
        tri_permutes = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ]

        face_to_ele = np.full((faces.shape[0]), 3, dtype=int)
        face_sort = np.sort(faces, axis=1)
        ele_tris = eles[:, tri_permutes]
        sorted_eletris = np.sort(ele_tris)

        for i, f in enumerate(face_sort):
            fequal = np.equal(f, sorted_eletris)
            e_idx = np.any(np.all(fequal, axis=2), axis=1)
            face_to_ele[i] = np.where(e_idx)[0]

        return face_to_ele

    def get_trimesh_surface(self):
        f_flat = self.faces.flatten()
        verts = self.nodes[f_flat]
        tris = np.arange(f_flat.size).reshape(-1, 3)
        pv_tris = np.hstack([np.full((tris.shape[0], 1), 3, dtype=int), tris])
        mesh = pv.PolyData(verts, pv_tris)
        return mesh

    def get_trimesh_volume(self):
        pv_eles = np.hstack([np.full((self.eles.shape[0], 1), 4, dtype=int), self.eles])
        celltypes = np.full(pv_eles.shape[0], fill_value=10, dtype=int)
        grid = pv.UnstructuredGrid(pv_eles, celltypes, self.nodes)
        return grid

    @property
    def n_eles(self):
        return self.eles.shape[0]

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    def __str__(self):
        return f"TetMesh {self.fn_base.as_posix()}\n" \
               f"#Nodes: {self.nodes.shape[0]}\n#Eles: {self.eles.shape[0]}\n#Faces: {None if self.faces is None else self.faces.shape[0]}"


class TetmeshPlotter:
    def __init__(self, tetmesh: Tetmesh, ele_vals: np.ndarray = None, scalar_name=None):

        self.tetmesh = tetmesh
        self.pvtet = tetmesh.get_trimesh_volume()

        assert ele_vals is None or ele_vals.size == tetmesh.n_eles or ele_vals.size == tetmesh.n_nodes
        if scalar_name is None:
            scalar_name = 'vals'
        if ele_vals is None:
            ele_vals = np.ones(tetmesh.n_eles)

        self.pvtet[scalar_name] = ele_vals  ## TODO: might have bugs if

        self.pltr = pv.Plotter()
        self.pltr.add_axes()
        self.colmode = 'YlGnBu'
        self.pltr.add_mesh(self.pvtet, scalars=self.pvtet.active_scalars_name, cmap=self.colmode, show_edges=True, line_width=.2, name='tetm')
        self.hardcoded_clim = None

        def clip_mesh(curr_box):
            clipped = self.get_clipped_mesh(curr_box)

            if self.hardcoded_clim is None:
                clim = (np.min(self.pvtet.active_scalars), np.max(self.pvtet.active_scalars))
                if self.colmode == 'RdBu':
                    clmax = max(np.abs(clim))
                    clim = (-clmax, clmax)
            else:
                clim = self.hardcoded_clim

            if clipped is not None:
                self.pltr.add_mesh(clipped, scalars=self.pvtet.active_scalars_name,
                                   clim=clim, cmap=self.colmode, show_edges=True, line_width=.2, name='tetm')
                self.clipped_mesh = clipped

        self.box = self.pltr.add_box_widget(clip_mesh, bounds=self.pvtet.bounds, rotation_enabled=False)
        self.callback_box = clip_mesh
        self.new_box = None

        # to be overwritten for saving the mesh
        self.clipped_mesh = None
        self.save_pth = SAVE_PTH_CLIP  # overwrite directly... UGLY

        def save_clipped():
            print('Saving...')

            if self.save_pth is None:
                print('Set a save path')
            tm = self.clipped_mesh
            if tm is None:
                print('Please clip the mesh first')
                return

            save_tetmesh(tm, self.save_pth)
            print('Saved Mesh')

        self.pltr.add_key_event(key='l', callback=save_clipped)

    def update(self):
        pd = pv.PolyData()
        self.box.GetPolyData(pd)
        self.callback_box(pd)

    def get_clipped_mesh(self, new_box=None):
        if new_box is not None:
            self.new_box = new_box

        previous_scalar = self.pvtet.active_scalars_name
        try:
            clipped = self.pvtet.clip_box(self.new_box.bounds, crinkle=True, invert=False)
        except AttributeError:
            print('AttributeError: self.new_box not initialised?')
            clipped = self.pvtet

        self.pvtet.set_active_scalars(previous_scalar)

        if clipped.n_cells == 0:
            return None

        return clipped

    def add_ctmesh_checkbox(self, ct_dir: Path):
        # lmao at python's memory management.
        # all these local object magically don't get garbage collected since they're referenced by the plotter I guess?
        contact_meshes = []
        for p in ct_dir.iterdir():
            print(p)
            assert p.is_file() and p.suffix in ['.ply', '.obj'], p.as_posix()
            contact_meshes.append(pv.PolyData(p.as_posix()))

        cm_acts = []
        for cm in contact_meshes:
            cm_acts.append(self.pltr.add_mesh(cm, style='wireframe', color='#d69e02'))

        self.pltr.add_checkbox_button_widget(lambda x: [cma.SetVisibility(x) for cma in cm_acts], value=True)

    def show(self):
        self.pltr.show()

    def update_pvtet(self, points, scalars):
        self.pvtet.points = points
        self.pltr.update_scalars(scalars, self.pvtet)
        self.update()


def remove_bad_tets(tetmesh: Tetmesh, show_change_plot=False):
    tm_pv = tetmesh.get_trimesh_volume()
    qual_mesh = tm_pv.compute_cell_quality(quality_measure='aspect_ratio')
    tets_to_keep = qual_mesh.cell_data['CellQuality'] < 10
    c_q = qual_mesh.extract_cells(tets_to_keep)
    c_notq = qual_mesh.extract_cells(np.logical_not(tets_to_keep))
    print('% of tets w. Q crit: ', np.count_nonzero(tets_to_keep) / np.size(tets_to_keep) * 100)

    if show_change_plot:
        p = pv.Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_mesh_clip_box(c_q)
        p.subplot(0, 1)
        p.add_mesh(c_notq)

        p.link_views()
        p.show()

    return c_q


def save_tetmesh(pv_tetmesh: pv.UnstructuredGrid, save_fn: Path):
    assert save_fn.parent.exists()
    nodes = pv_tetmesh.points
    eles = pv_tetmesh.cells
    face_surf = pv_tetmesh.extract_surface()
    origninal_pt_ids = face_surf['vtkOriginalPointIds']
    faces = face_surf.faces.reshape(-1, 4)[:, 1:]

    original_faces = origninal_pt_ids[faces]

    with open(save_fn.as_posix() + '.node', 'w') as nf:
        meta_str = f"{pv_tetmesh.n_points} 3 0 0\n"
        nf.write(meta_str)
        node_str_list = [f"{i}\t{n[0]}\t{n[1]}\t{n[2]}\n" for i, n in enumerate(nodes)]
        nf.writelines(node_str_list)

    with open(save_fn.as_posix() + '.ele', 'w') as ef:
        meta_str = f"{pv_tetmesh.n_cells} 4 0\n"
        ef.write(meta_str)
        ele_str_list = [f"{i}\t{e[1]}\t{e[2]}\t{e[3]}\t{e[4]}\n" for i, e in enumerate(eles.reshape(-1, 5))]
        ef.writelines(ele_str_list)

    with open(save_fn.as_posix() + '.face', 'w') as ff:
        meta_str = f"{face_surf.n_faces} 0\n"
        ff.write(meta_str)
        face_str_list = [f"{i}\t{f[0]}\t{f[1]}\t{f[2]}\n" for i, f in enumerate(original_faces)]
        ff.writelines(face_str_list)

    face_surf.save(save_fn.as_posix() + ".ply")

    print('Saved Tetmesh and Surface to ', save_fn)
    return True


def _main_plot(mesh_fn, teeth_dir=None, save_fn=None):
    tetmesh = Tetmesh(mesh_fn)
    print(tetmesh)
    tm_pv = tetmesh.get_trimesh_volume()

    size_mesh = tm_pv.compute_cell_sizes()
    vols = size_mesh.cell_data['Volume']
    qual_mesh = tm_pv.compute_cell_quality(quality_measure='aspect_ratio')
    quals = qual_mesh.cell_data['CellQuality']

    log_volume = np.log10(vols)
    pltr = TetmeshPlotter(tetmesh, ele_vals=log_volume, scalar_name='(Log) Element volume')

    if teeth_dir is not None and teeth_dir.exists():
        pltr.add_ctmesh_checkbox(teeth_dir)

    v_filt = vols < 0.2
    #q_filt = np.logical_and(4 < quals, quals < 25)
    q_filt = 10 < quals

    c_v = size_mesh.extract_cells(v_filt)
    c_q = qual_mesh.extract_cells(q_filt)

    pltr.pltr.add_mesh(c_q, style='wireframe', scalars='CellQuality')

    print('% of tets w. V crit: ', np.count_nonzero(v_filt) / np.size(vols) * 100)
    print('% of tets w. Q crit: ', np.count_nonzero(q_filt) / np.size(quals) * 100)

    pltr.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_file', type=Path)

    args = parser.parse_args()

    assert args.node_file.exists()

    _main_plot(args.node_file.with_suffix(''))
