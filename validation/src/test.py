import pymeshlab as ml
ms = ml.MeshSet()

# 1) See the exact parameter list for the remesh filter on your build:
print(ms.print_filter_parameter_list('meshing_isotropic_explicit_remeshing'))

# 2) Find the correct selection filters by listing all filters and grepping:
flist = ms.print_filter_list()
print(flist)
# Then search the text for:
#   "selection", "border", "invert", "taubin", "smooth"
