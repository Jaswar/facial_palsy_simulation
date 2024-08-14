import numpy as np
from tqdm import tqdm


# tried using libraries like pywavefront, but it seems like it doesn't support uv coordinates properly
class ObjParser(object):

    def __init__(self):
        pass

    def parse(self, path, progress_bar=False, mrgb_only=False):
        vertices = []
        faces = []
        normals = []
        uv_coords = []
        rgb_values = []
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        for line in tqdm(lines, disable=not progress_bar):
            if line.startswith('v ') and not mrgb_only:
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt ') and not mrgb_only:
                uv_coords.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vn ') and not mrgb_only:
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('#MRGB'):
                rgb_values.extend(self.__parse_mrgb(line))
            elif line.startswith('f ') and not mrgb_only:
                faces.append(self.__parse_face(line))
        return np.array(vertices), np.array(faces), np.array(normals), np.array(uv_coords), np.array(rgb_values, dtype=np.uint8)


    def __parse_face(self, face_line):
        split = face_line.strip().split(' ')[1:]
        face = [[int(v) if v != '' else -1 for v in vertex.split('/')] for vertex in split if vertex != '']
        return face
    
    def __parse_mrgb(self, mrgb_line):
        mrgb_line = mrgb_line.strip().split(' ')[1]
        mrgb_line = mrgb_line[:8*64]
        rgb_values = [mrgb_line[i:i+8] for i in range(0, len(mrgb_line), 8)]
        rgb_values = [[int(color[i:i+2], 16) for i in range(2, 8, 2)] for color in rgb_values]
        # rgb_values = [[color[0] & color[i] for i in range(1, 4)] for color in rgb_values]
        return rgb_values
