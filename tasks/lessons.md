
## 2026-09-02: rendering shipped with the wrong attitude
- What happened: `Scene.pose` sliced the quaternion from the canonical 13-vector at index 3,
  which is the velocity (layout: position, velocity, attitude, rates). The videos showed a
  tailsitter stopping mid-air without pitching and a conventional aircraft "falling along the
  ground"; the user caught it, not the tests.
- Rule: never slice a packed vector by assumption; read the packer (or use its unpacker) first.
- Rule: a visual pipeline gets a numerical readback test before anyone looks at a video: pose
  known attitudes (right bank, nose up, yaw east) and assert where the renderer put the wings
  and the nose. Frames are for taste; readbacks are for correctness.
- Rule: MuJoCo aborts the process without an OpenGL context. Gate rendering tests on an
  explicit signal (MUJOCO_GL, DISPLAY, CASCADE_RENDER_TESTS) and skip under CI on every OS;
  GitHub's macOS runners have no window server either.
