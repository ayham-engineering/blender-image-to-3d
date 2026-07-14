import bpy
import math

def add_cube(name, dims, pos, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (dims[0]/2, dims[1]/2, dims[2]/2)
    obj.rotation_euler = rot
    return obj

def add_cylinder(name, dims, pos, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=dims[0]/2, depth=dims[2], location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (1, dims[1]/dims[0], 1)
    obj.rotation_euler = rot
    return obj

# Reference: a two-plank folding lawn chair (Kentucky stick chair style).
# The backrest is tall and tilted back with many vertical cords.
# The seat plank tilts down toward the front. The whole thing rests on
# the ground via the front leg tips and rear leg tips forming a wide stance.
# Backrest and seat are two flat panels crossing at a low pivot.

# Key corrections:
# - Backrest tilts back MORE (~0.55 rad) and is taller/wider cord panel
# - Pivot is low and forward; seat plank extends forward from pivot
# - Front "leg" is really the front end of the seat plank meeting ground
# - Wider frame rails, denser cords, clear crossbars
# - Everything scaled to ground level

parts = [
    # Backrest rails: tilted well back, tall, meeting pivot low
    ("back_frame_left_rail", "cube", [0.07,0.06,1.35], [-0.30,0.42,0.72], [0.55,0,0]),
    ("back_frame_right_rail", "cube", [0.07,0.06,1.35], [0.30,0.42,0.72], [0.55,0,0]),
    ("back_top_crossbar", "cube", [0.68,0.07,0.07], [0.0,-0.20,1.35], [0.55,0,0]),
    ("back_bottom_crossbar", "cube", [0.68,0.07,0.07], [0.0,0.72,0.20], [0.55,0,0]),
    ("back_cords", "cube", [0.54,0.015,1.20], [0.0,0.42,0.73], [0.55,0,0]),
    # Seat: plank tilts down to front from pivot
    ("seat_cords", "cube", [0.54,0.70,0.03], [0.0,0.95,0.25], [0.18,0,0]),
    ("seat_side_left", "cube", [0.06,0.78,0.06], [-0.29,0.92,0.27], [0.18,0,0]),
    ("seat_side_right", "cube", [0.06,0.78,0.06], [0.29,0.92,0.27], [0.18,0,0]),
    ("seat_front_crossbar", "cube", [0.66,0.06,0.06], [0.0,1.28,0.18], [0,0,0]),
    ("seat_pivot_crossbar", "cube", [0.66,0.06,0.06], [0.0,0.68,0.32], [0,0,0]),
    # Front legs: extend down-forward from front of seat to ground
    ("front_leg_left", "cube", [0.06,0.06,0.55], [-0.29,1.20,-0.05], [-0.30,0,0]),
    ("front_leg_right", "cube", [0.06,0.06,0.55], [0.29,1.20,-0.05], [-0.30,0,0]),
    ("front_foot_crossbar", "cube", [0.64,0.06,0.06], [0.0,1.32,-0.24], [0,0,0]),
    # Rear legs: splay backward from pivot down to ground
    ("rear_leg_left", "cube", [0.06,0.06,0.72], [-0.29,0.45,-0.02], [0.70,0,0]),
    ("rear_leg_right", "cube", [0.06,0.06,0.72], [0.29,0.45,-0.02], [0.70,0,0]),
    ("rear_foot_crossbar", "cube", [0.64,0.06,0.06], [0.0,0.66,-0.32], [0,0,0]),
    # Pivot dowel through both frames
    ("pivot_bolt", "cylinder", [0.07,0.07,0.70], [0.0,0.70,0.30], [0,1.5708,0]),
]

created = []
for name, prim, dims, pos, rot in parts:
    if prim == "cube":
        obj = add_cube(name, dims, pos, rot)
    else:
        obj = add_cylinder(name, dims, pos, rot)
    created.append(obj)

# apply overall scale
for obj in created:
    obj.location = tuple(c*1.5 for c in obj.location)
    obj.scale = tuple(s*1.5 for s in obj.scale)