"""Optional keyboard input and body-command scaling for policy replay."""


def held_axes(keys):
    """Normalized body +X forward, +Y left, +yaw counterclockwise; opposites cancel."""
    if "space" in keys:
        return (0.0, 0.0, 0.0)
    scale = 0.3 if "shift" in keys else 1.0
    return tuple(scale * (int(positive in keys) - int(negative in keys))
                 for positive, negative in (("w", "s"), ("a", "d"), ("q", "e")))


def scale_axes(axes, limits):
    """Map normalized axes to directional bounds in m/s, m/s, rad/s."""
    return tuple(0.0 if axis == 0 else
                 min(1.0, max(-1.0, axis)) * max(0.0, high if axis > 0 else -low)
                 for axis, (low, high) in zip(axes, limits))


class KeyboardInput:
    """A pygame 2 window, initialized and polled on the replay thread."""

    def __init__(self):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Manual control requires pygame 2. Install it with: "
                               "python -m pip install pygame") from error
        self.pg = pygame
        self.quit = False
        self.wait_for_release = True
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((500, 220))
            pygame.display.set_caption("Policy replay keyboard control")
            self.font = pygame.font.Font(None, 25)
            self.draw((0.0, 0.0, 0.0))
        except Exception:
            self.close()
            raise
        print("Manual control: focus the keyboard input window; keep Genesis visible. "
              "W/S vx, A/D vy, Q/E yaw; Shift slow, Space stop, Esc quit.", flush=True)

    def poll(self):
        """Return three normalized axes and quit; never store movement keydowns."""
        pg = self.pg
        for event in pg.event.get():
            if event.type in (pg.QUIT, pg.WINDOWCLOSE):
                self.quit = True
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.quit = True
            elif event.type == pg.WINDOWFOCUSLOST:
                self.wait_for_release = True
        if not pg.key.get_focused():
            self.wait_for_release = True
            return (0.0, 0.0, 0.0), self.quit
        pressed = pg.key.get_pressed()
        if pressed[pg.K_ESCAPE]:
            self.quit = True
        keys = {name for name, code in (("w", pg.K_w), ("s", pg.K_s),
                ("a", pg.K_a), ("d", pg.K_d), ("q", pg.K_q), ("e", pg.K_e),
                ("space", pg.K_SPACE), ("shift", pg.K_LSHIFT), ("shift", pg.K_RSHIFT))
                if pressed[code]}
        # Require release after focus loss, even if focus returned between ticks.
        if self.wait_for_release:
            self.wait_for_release = bool(keys)
            return (0.0, 0.0, 0.0), self.quit
        return held_axes(keys), self.quit

    def draw(self, command):
        self.screen.fill((25, 28, 34))
        lines = ("Keep this window focused to send commands.",
                 "W/S: forward/reverse     A/D: left/right",
                 "Q/E: CCW/CW yaw     Shift: x0.3",
                 "Space: stop     Esc or close either window: quit",
                 f"vx {command[0]:+.2f} m/s     vy {command[1]:+.2f} m/s",
                 f"yaw rate {command[2]:+.2f} rad/s")
        for row, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (230, 235, 240)), (12, 12 + row * 32))
        self.pg.display.flip()

    def close(self):
        self.pg.display.quit()
        self.pg.font.quit()
