from pfc3d_postprocess import grain_misorientation_map

out = grain_misorientation_map(
    "/home/ubuntu/PycharmProjects/PythonProject/VTK/grain_data.csv",
    like_vtk="/home/ubuntu/PycharmProjects/PythonProject/VTK/fields_002000.vtk",
    ref_grain=0,
    z_frac=0.5,
    angle_unit="rad",
    out_path="/home/ubuntu/PycharmProjects/PythonProject/post/misori_map.png",
)
print("wrote", out)