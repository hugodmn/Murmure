from dataclasses import dataclass

@dataclass
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
    


@dataclass
class AudioSegment():

    def __init__(
            self,
            speaker_id : int,
            starting_time : int,
            ending_time : int, 
            transcription : str = None

    ):

        self.speaker_id = speaker_id
        self.starting_time = starting_time
        self.ending_time = ending_time 
        self.transcription = transcription


    # def __str__(self):
    #     return f"AudioSegment(speaker_id={self.speaker_id}, starting_time={self.starting_time}s, ending_time={self.ending_time})"

    def __repr__(self):
        return f"AudioSegment(speaker_id={self.speaker_id}, transcription={self.transcription}, starting_time={self.starting_time}s, ending_time={self.ending_time})"
