import numpy as np

# tried using libraries like pywavefront, but it seems like it doesn't support uv coordinates properly
class ObjParser(object):

    def __init__(self):
        pass

    def parse(self, path):
        vertices = []
        faces = []
        normals = []
        uv_coords = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('vt '):
                    uv_coords.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('vn '):
                    normals.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append(self.__parse_face(line))
        return np.array(vertices), np.array(faces), np.array(normals), np.array(uv_coords)


    def __parse_face(self, face_line):
        split = face_line.strip().split(' ')[1:]
        face = [[int(v) for v in vertex.split('/')] for vertex in split if vertex != '']
        return face