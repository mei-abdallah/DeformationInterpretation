import os

class CheckPoint:
    def __init__(self, dirname:str, strtime:str) -> None:
        self.dirname = dirname
        self.strtime = strtime
        self.createdirs()
        
    def __str__(self) -> str:
        return f'{self.dirname}/{self.strtime}'
    
    def createdirs(self):
        os.makedirs(str(self), exist_ok=True)