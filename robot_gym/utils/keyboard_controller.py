"""Optional shared pygame command window; no hardware transport."""


class KeyboardController:
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.window = pygame.display.set_mode((620, 100))
        pygame.display.set_caption("Locomotion: W/S forward, A/D lateral, Q/E yaw, Space stop, Esc quit")
        self.clock = pygame.time.Clock()

    def command(self):
        p = self.pg
        for event in p.event.get():
            if event.type == p.QUIT:
                raise KeyboardInterrupt
        keys = p.key.get_pressed()
        if keys[p.K_ESCAPE]:
            raise KeyboardInterrupt
        self.clock.tick(50)
        if keys[p.K_SPACE]:
            return (0., 0., 0.)
        return (.5 * (keys[p.K_w] - keys[p.K_s]), .3 * (keys[p.K_a] - keys[p.K_d]),
                .8 * (keys[p.K_q] - keys[p.K_e]))

    def close(self):
        self.pg.quit()
