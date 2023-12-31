#!/usr/bin/env python3

import pygame
from pygame.locals import DOUBLEBUF

class Display(object):
  def __init__(self, W, H):
    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size()).convert()
    print(self.surface)

  def paint(self, img):
    pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [2,1,0]]) # BGR > RGB
    self.screen.blit(self.surface, (0,0))
    pygame.display.flip()