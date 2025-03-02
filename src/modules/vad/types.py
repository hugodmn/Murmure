

class VADSegmentOutput():
    def __init__(self, 
                 idx : int, 
                 silence_removed : int,
                 start : int, 
                 end : int):
        
        self.idx = idx 
        self.silence_removed = silence_removed
        self.start = start 
        self.end = end 

    
    def __repr__(self):
        return f"VADSegmentOutput(id={self.idx}, start={self.start}, end={self.end}, silence_removed={self.silence_removed})"