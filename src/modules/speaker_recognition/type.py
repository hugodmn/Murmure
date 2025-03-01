class AudioSegment():

    def __init__(
            self,
            speaker_id : int,
            starting_time : int,
            ending_time : int, 

    ):

        self.speaker_id = speaker_id
        self.starting_time = starting_time
        self.ending_time = ending_time 