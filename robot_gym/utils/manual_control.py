"""Optional keyboard input and body-command scaling for policy replay."""

from math import pi


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


def adjust_limits(limits, axis, delta):
    """Change both directional speed magnitudes; floor at zero, with no training cap."""
    result = list(limits)
    low, high = result[axis]
    result[axis] = (-max(0.0, round(max(0.0, -low) + delta, 10)),
                   max(0.0, round(max(0.0, high) + delta, 10)))
    return tuple(result)


class KeyboardInput:
    """A pygame 2 window, initialized and polled on the replay thread."""

    def __init__(self, limits):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Manual control requires pygame 2. Install it with: "
                               "python -m pip install pygame") from error
        self.pg = pygame
        self.quit = False
        self.wait_for_release = True
        self.keys = set()
        self.limits = tuple(tuple(bounds) for bounds in limits)
        self.selected_axis = 0
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((640, 500))
            pygame.display.set_caption("Policy replay keyboard control")
            self.font = pygame.font.Font(None, 25)
            self.minus_button = pygame.Rect(480, 382, 54, 40)
            self.plus_button = pygame.Rect(554, 382, 54, 40)
            self.draw((0.0, 0.0, 0.0))
        except Exception:
            self.close()
            raise
        print("Manual control: focus the keyboard input window; keep Genesis visible. "
              "W/S vx, A/D vy, Q/E yaw; Shift slow, Space stop, Esc quit. "
              "Tab selects vx/vy/yaw; +/- adjusts both bounds by 0.1 (no training cap).", flush=True)

    def poll(self):
        """Return three normalized axes and quit; never store movement keydowns."""
        pg = self.pg
        events = pg.event.get()
        focused = pg.key.get_focused()
        self.keys = set()
        for event in events:
            if event.type in (pg.QUIT, pg.WINDOWCLOSE):
                self.quit = True
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.quit = True
            elif event.type == pg.WINDOWFOCUSLOST:
                self.wait_for_release = True
            elif event.type == pg.KEYDOWN and focused and not getattr(event, "repeat", False):
                if event.key == pg.K_TAB:
                    self.selected_axis = (self.selected_axis + 1) % 3
                elif event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    self.limits = adjust_limits(self.limits, self.selected_axis, 0.1)
                elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    self.limits = adjust_limits(self.limits, self.selected_axis, -0.1)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and focused:
                if self.plus_button.collidepoint(event.pos):
                    self.limits = adjust_limits(self.limits, self.selected_axis, 0.1)
                elif self.minus_button.collidepoint(event.pos):
                    self.limits = adjust_limits(self.limits, self.selected_axis, -0.1)
        if not focused:
            self.wait_for_release = True
            return (0.0, 0.0, 0.0), self.quit
        pressed = pg.key.get_pressed()
        if pressed[pg.K_ESCAPE]:
            self.quit = True
        self.keys = {name for name, code in (("w", pg.K_w), ("s", pg.K_s),
                ("a", pg.K_a), ("d", pg.K_d), ("q", pg.K_q), ("e", pg.K_e),
                ("space", pg.K_SPACE), ("shift", pg.K_LSHIFT), ("shift", pg.K_RSHIFT))
                if pressed[code]}
        # Require release after focus loss, even if focus returned between ticks.
        if self.wait_for_release:
            self.wait_for_release = bool(self.keys)
            return (0.0, 0.0, 0.0), self.quit
        return held_axes(self.keys), self.quit

    def _arrow(self, key, center, direction, label):
        """Draw held keys, including canceled inputs or inputs overridden by Space."""
        pg = self.pg
        color = (250, 177, 55) if key in self.keys else (85, 94, 108)
        x, y = center
        if direction is None:
            # Quarter turns: Q ends pointing left (CCW), E ends pointing right (CW).
            pg.draw.arc(self.screen, color, (x - 28, y - 28, 56, 56),
                        0 if key == "q" else pi / 2, pi / 2 if key == "q" else pi, 5)
            sign = -1 if key == "q" else 1
            points = ((x + sign * 9, y - 28), (x - sign * 5, y - 35), (x - sign * 5, y - 21))
        else:
            dx, dy = direction
            pg.draw.line(self.screen, color, (x - dx * 18, y - dy * 18), (x + dx * 7, y + dy * 7), 5)
            points = ((x + dx * 21, y + dy * 21),
                      (x + dx * 5 - dy * 10, y + dy * 5 + dx * 10),
                      (x + dx * 5 + dy * 10, y + dy * 5 - dx * 10))
        pg.draw.polygon(self.screen, color, points)
        text = self.font.render(label, True, color)
        self.screen.blit(text, text.get_rect(center=(x, y + 42)))

    def _text(self, text, position, color=(230, 235, 240)):
        self.screen.blit(self.font.render(text, True, color), position)

    def draw(self, command):
        pg = self.pg
        self.screen.fill((25, 28, 34))
        self._text("Keep this window focused to send commands.", (18, 14))
        self._text("Orange = pressed. Opposite keys cancel; Space overrides movement.", (18, 42), (160, 170, 185))
        for key, center, direction, label in (
            ("w", (320, 94), (0, -1), "W  +vx"),
            ("s", (320, 224), (0, 1), "S  -vx"),
            ("a", (230, 159), (-1, 0), "A  +vy"),
            ("d", (410, 159), (1, 0), "D  -vy"),
            ("q", (100, 159), None, "Q  +yaw (CCW)"),
            ("e", (540, 159), None, "E  -yaw (CW)"),
        ):
            self._arrow(key, center, direction, label)
        for key, text, x in (("shift", "Shift: x0.3", 18), ("space", "Space: stop", 185)):
            self._text(text, (x, 285), (250, 177, 55) if key in self.keys else (85, 94, 108))
        self._text("Esc / close: quit", (440, 285))
        self._text("Tab: select axis    +/-: change both limits by 0.1", (18, 320))
        for axis, (name, units) in enumerate((("vx", "m/s"), ("vy", "m/s"), ("yaw", "rad/s"))):
            low, high = self.limits[axis]
            rect = pg.Rect(18, 347 + axis * 39, 432, 34)
            selected = axis == self.selected_axis
            pg.draw.rect(self.screen, (61, 47, 30) if selected else (35, 39, 47), rect, border_radius=5)
            if selected:
                pg.draw.rect(self.screen, (250, 177, 55), rect, width=1, border_radius=5)
            self._text(f"{name} limits: {low:+.2f} .. {high:+.2f} {units}", (28, rect.y + 6))
        for rect, text in ((self.minus_button, "-"), (self.plus_button, "+")):
            pg.draw.rect(self.screen, (61, 47, 30), rect, border_radius=5)
            pg.draw.rect(self.screen, (250, 177, 55), rect, width=1, border_radius=5)
            label = self.font.render(text, True, (250, 177, 55))
            self.screen.blit(label, label.get_rect(center=rect.center))
        self._text(f"Adjust {('vx', 'vy', 'yaw')[self.selected_axis]}", (485, 351))
        self._text(f"Now: vx {command[0]:+.2f} m/s    vy {command[1]:+.2f} m/s    yaw {command[2]:+.2f} rad/s", (18, 474))
        pg.display.flip()

    def close(self):
        self.pg.display.quit()
        self.pg.font.quit()
