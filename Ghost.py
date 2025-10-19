import math
import random

class Ghost :
    def __init__(self, x, y, cx, cy, speed):

        self.x, self.y = x, y
        self.cx, self.cy = cx, cy
        self.speed = speed
        self.flipped = False
        self.symbols = []

        if (self.x > cx):
            self.flipped = True

        for i in range(random.randint(1,5)):
            self.addSymbol(random.randint(0,3))

    def move(self):

        # Direction vector (from ghost to center)
        dx = self.cx - self.x
        dy = self.cy - self.y

        # Distance to center
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0:
            return  # already at center

        # Normalize direction and move
        dx /= dist
        dy /= dist

        self.x += dx * self.speed
        self.y += dy * self.speed
    
    def checkMatch(self,predict):
        if(self.symbols[0] == predict):
            del self.symbols[0]
        
    def addSymbol(self, num):
        self.symbols.append(num)
        
    def isDead(self):
        return len(self.symbols) == 0
        

        
            
            
    
