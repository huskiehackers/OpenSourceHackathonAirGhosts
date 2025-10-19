import math
import random

class Ghost :
    
    def __init__(self, x, y, cx, cy, speed):

        self.x, self.y = x, y
        self.cx, self.cy = cx, cy
        self.speed = speed
        self.symbols = []
        
        for(i in range(5)):
            addSymbol(self,random.randint(0,10)

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
    
    def symbol(self,predict):
        if(self.symbols[0] == predict):
            del self.symbols[0]
        return len(self.symbols) == 0
        
        
    def addSymbol(self, num)
        if(num >= 0 && num <= 3):
            self.symbols.append(num)
        
            
            
    
