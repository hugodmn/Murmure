


class TranscriptionOutput():

    def __init__(self,
                 idx : int,
                 transcription : str,
                 start : float,
                 end : float,
                 speaker_id : int = None
                 ):
        
        self.idx = idx
        self.transcription = transcription
        self.start = start
        self.end = end 

        self.speaker_id = speaker_id

    def __repr__(self,):
        return f"AudioSegment(id={self.idx}, speaker_id={self.speaker_id}, transcription={self.transcription}, start={self.start}s, end={self.end}s)"