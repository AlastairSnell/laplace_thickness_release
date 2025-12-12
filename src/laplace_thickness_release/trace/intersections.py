import numpy as np

def segment_triangle_intersection(p0, p1, tri_vertices):
    """
    Returns (hit_bool, intersection_point, tval)
    Möller–Trumbore intersection with segment p0->p1
    """
    v0, v1, v2= tri_vertices
    dir_seg= p1- p0
    epsilon=1e-14

    edge1= v1- v0
    edge2= v2- v0
    h= np.cross(dir_seg, edge2)
    a= np.dot(edge1,h)
    if abs(a)< epsilon:
        return (False,None,999.9)
    f=1.0/a
    s= p0- v0
    u= f* np.dot(s,h)
    if u<0.0 or u>1.0:
        return (False,None,999.9)
    q= np.cross(s,edge1)
    v= f* np.dot(dir_seg,q)
    if v<0.0 or (u+v)>1.0:
        return (False,None,999.9)
    t= f* np.dot(edge2,q)
    if t<0.0 or t>1.0:
        return (False,None,999.9)
    hit_point= p0+ t* dir_seg
    return (True, hit_point,t)

def find_exit_intersection(p0, p1, all_triangles):
    best_t= 999.9
    best_point= None
    for tri in all_triangles:
        tri_verts= tri['vertices']
        hit, xint, tval= segment_triangle_intersection(p0,p1,tri_verts)
        if hit and tval< best_t:
            best_t= tval
            best_point= xint
    return best_point