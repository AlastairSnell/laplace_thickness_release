import numpy as np

def preprocess_triangles(triangles):
    
    for tri in triangles:
        verts = tri['vertices']
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        cross_prod = np.cross(e1, e2)
        area = 0.5*np.linalg.norm(cross_prod)
        normal_geom = cross_prod / (2.0*area)
        if np.dot(normal_geom, tri['normal'])<0:
            normal_geom = -normal_geom
        tri['area']     = area
        tri['normal']   = normal_geom
        tri['centroid'] = np.mean(verts, axis=0)