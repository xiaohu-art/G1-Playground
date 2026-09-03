import glfw
import mujoco
import numpy as np

from g1_playground.simulation.mujoco_depth import (
    CAMERA_NAME,
    MUJOCO_DEPTH_PATH,
    DepthFrameWriter,
    depth_buffer_to_meters,
)

DEPTH_DISPLAY_SCALE = 2
DEPTH_DISPLAY_MARGIN = 10


def depth_display(depth_m) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= 0.25) & (depth < 3.0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = ((3.0 - depth[valid]) / (3.0 - 0.25) * 255.0).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    return np.repeat(np.repeat(image, DEPTH_DISPLAY_SCALE, axis=0), DEPTH_DISPLAY_SCALE, axis=1)


class MujocoViewer:
    """One-context MuJoCo window with an optional depth picture-in-picture."""

    def __init__(
        self,
        model,
        data,
        *,
        depth: bool = False,
        camera_name: str = CAMERA_NAME,
        depth_path: str = MUJOCO_DEPTH_PATH,
        width: int = 1280,
        height: int = 720,
    ):
        self.model = model
        self.data = data
        self.window = None
        self.context = None
        self.depth_writer = None
        self.depth_image = None
        self._left_pressed = False
        self._right_pressed = False
        self._middle_pressed = False
        self._last_cursor = (0.0, 0.0)

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        try:
            self.window = glfw.create_window(width, height, "G1 Playground", None, None)
            if self.window is None:
                raise RuntimeError("Failed to create MuJoCo viewer window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            self.option = mujoco.MjvOption()
            self.perturb = mujoco.MjvPerturb()
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self.cam)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            self.context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR

            self.depth_camera = None
            self.depth_scene = None
            self.depth_viewport = None
            self.depth_buffer = None
            if depth:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if camera_id < 0:
                    raise ValueError(f"MuJoCo model has no camera named {camera_name!r}")
                camera_width, camera_height = np.asarray(model.cam_resolution[camera_id], dtype=int)
                if camera_width <= 0 or camera_height <= 0:
                    raise ValueError(f"MuJoCo camera {camera_name!r} has no image resolution")
                if camera_width > model.vis.global_.offwidth or camera_height > model.vis.global_.offheight:
                    raise ValueError(
                        f"MuJoCo camera resolution {camera_width}x{camera_height} exceeds offscreen buffer"
                    )
                self.depth_camera = mujoco.MjvCamera()
                self.depth_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.depth_camera.fixedcamid = camera_id
                self.depth_scene = mujoco.MjvScene(model, maxgeom=10000)
                self.depth_scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
                self.depth_scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
                self.depth_viewport = mujoco.MjrRect(0, 0, int(camera_width), int(camera_height))
                self.depth_buffer = np.empty((camera_height, camera_width), dtype=np.float32)
                self.depth_writer = DepthFrameWriter(int(camera_height), int(camera_width), depth_path)

            glfw.set_key_callback(self.window, self._key_callback)
            glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
        except BaseException:
            self.close()
            raise

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def render(self, *, render_depth: bool = False, timestamp: float | None = None) -> None:
        if not self.is_running():
            return
        glfw.make_context_current(self.window)

        if render_depth and self.depth_writer is not None:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.option,
                None,
                self.depth_camera,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.depth_scene,
            )
            mujoco.mjr_render(self.depth_viewport, self.depth_scene, self.context)
            mujoco.mjr_readPixels(None, self.depth_buffer, self.depth_viewport, self.context)
            depth_m = depth_buffer_to_meters(np.flipud(self.depth_buffer), self.model)
            self.depth_writer.write(depth_m, timestamp)
            self.depth_image = depth_display(depth_m)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if self.depth_image is not None:
            image_height, image_width = self.depth_image.shape[:2]
            inset = mujoco.MjrRect(
                DEPTH_DISPLAY_MARGIN,
                max(height - image_height - DEPTH_DISPLAY_MARGIN, 0),
                min(image_width, width),
                min(image_height, height),
            )
            mujoco.mjr_rectangle(inset, 0.0, 0.0, 0.0, 1.0)
            pixels = np.ascontiguousarray(np.flipud(self.depth_image)).reshape(-1)
            mujoco.mjr_drawPixels(pixels, None, inset, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.depth_writer is not None:
            self.depth_writer.close()
            self.depth_writer = None
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def _key_callback(self, window, key, _scancode, action, _mods) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def _mouse_button_callback(self, window, button, action, _mods) -> None:
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._middle_pressed = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _cursor_callback(self, window, xpos, ypos) -> None:
        if not (self._left_pressed or self._right_pressed or self._middle_pressed):
            return
        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        _, height = glfw.get_framebuffer_size(window)
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model,
            action,
            (xpos - last_x) / height,
            (ypos - last_y) / height,
            self.scene,
            self.cam,
        )

    def _scroll_callback(self, _window, _x_offset, y_offset) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            self.scene,
            self.cam,
        )
