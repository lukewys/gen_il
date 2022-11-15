import glob
from torch.utils.data import Dataset
import numpy as np
import torch
import pretty_midi
import logging
import os


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    # https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def single_channel_pianoroll_save_to_midi(pianoroll, output_dir, save_wav=True, suffix='',
                                          min_pitch=34, max_pitch=81):
    # timidity only supports path without ".". it will replace "." with "_".
    tempo = 80
    frames_per_quarter = 4
    program = 0
    fs = int(tempo * frames_per_quarter / 60)
    pianoroll = pianoroll[:, 0, :, :]  # collapse channel dim
    batch_size = pianoroll.shape[0]
    length = pianoroll.shape[1]
    pianoroll = np.concatenate([np.zeros((batch_size, length, min_pitch - 1)),
                                pianoroll,
                                np.zeros((batch_size, length, 128 - max_pitch))], axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    suffix = '_' + suffix if suffix else ''
    # data_seq: numpy array, [batch_size, t*4(SATB)]
    for i in range(pianoroll.shape[0]):
        pianoroll_single = pianoroll[i].T
        midi_data = piano_roll_to_pretty_midi(pianoroll_single * 100, fs=fs, program=program)
        save_path = os.path.join(output_dir, f'{i}{suffix}.mid')
        midi_data.write(save_path)
        if save_wav:
            midi_to_wav(save_path)


def midi_to_wav(midi_path):
    midi_path = os.path.abspath(midi_path)
    wav_path = midi_path.replace('.mid', '.wav')
    mp3_path = midi_path.replace('.mid', '.mp3')
    os.system(f'timidity --quiet -T 120 --output-24bit -Ow "{midi_path}"')
    os.system(f'ffmpeg -v quiet -i "{wav_path}" -ar 44100 "{mp3_path}"')
    os.system(f'rm "{wav_path}"')


class PitchOutOfEncodeRangeError(Exception):
    """Exception for when pitch of note is out of encodings range."""
    pass


class PianorollEncoderDecoder(object):
    """Encodes list/array format piece_pitch_frame_sequence into pianorolls and decodes into midi."""

    qpm = 120
    # Oboe, English horn, clarinet, bassoon, sounds better on timidity.
    programs = [69, 70, 72, 71]

    def __init__(self,
                 shortest_duration=0.125,
                 min_pitch=36,
                 max_pitch=81,
                 separate_instruments=True,
                 num_instruments=None,
                 quantization_level=None):
        assert num_instruments is not None
        self.shortest_duration = shortest_duration
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.separate_instruments = separate_instruments
        self.num_instruments = num_instruments
        self.quantization_level = quantization_level
        if quantization_level is None:
            quantization_level = self.shortest_duration

    def encode(self, sequence):
        """Encode sequence into pianoroll."""
        # Sequence can either be a 2D numpy array or a list of lists.
        if (isinstance(sequence, np.ndarray) and sequence.ndim == 2) or (
                isinstance(sequence, list) and
                isinstance(sequence[0], (list, tuple))):
            # If sequence is an numpy array should have shape (time, output_channel).
            if (isinstance(sequence, np.ndarray) and
                    sequence.shape[-1] != self.num_instruments):
                raise ValueError('Last dim of sequence should equal output_channel.')
            if isinstance(sequence, np.ndarray) and not self.separate_instruments:
                raise ValueError('Only use numpy array if instruments are separated.')
            sequence = list(sequence)
            return self.encode_list_of_lists(sequence)
        else:
            raise TypeError('Type %s not yet supported.' % type(sequence))

    def encode_list_of_lists(self, sequence):
        """Encode 2d array or list of lists of midi note numbers into pianoroll."""
        # step_size larger than 1 means some notes will be skipped over.
        step_size = self.quantization_level / self.shortest_duration
        if not step_size.is_integer():
            raise ValueError(
                'quantization %r should be multiple of shortest_duration %r.' %
                (self.quantization_level, self.shortest_duration))
        step_size = int(step_size)

        if not (len(sequence) / step_size).is_integer():
            raise ValueError('step_size %r should fully divide length of seq %r.' %
                             (step_size, len(sequence)))
        tt = int(len(sequence) / step_size)
        pp = self.max_pitch - self.min_pitch + 1
        if self.separate_instruments:
            roll = np.zeros((tt, pp, self.num_instruments))
        else:
            roll = np.zeros((tt, pp, 1))
        for raw_t, chord in enumerate(sequence):
            # Only takes time steps that are on the quantization grid.
            if raw_t % step_size != 0:
                continue
            t = int(raw_t / step_size)
            for i in range(self.num_instruments):
                if i > len(chord):
                    # Some instruments are silence in this time step.
                    if self.separate_instruments:
                        raise ValueError(
                            'If instruments are separated must have all encoded.')
                    continue
                pitch = chord[i]
                # Silences are sometimes encoded as NaN when instruments are separated.
                if np.isnan(pitch):
                    continue
                if pitch > self.max_pitch or pitch < self.min_pitch:
                    raise PitchOutOfEncodeRangeError(
                        '%r is out of specified range [%r, %r].' % (pitch, self.min_pitch,
                                                                    self.max_pitch))
                p = pitch - self.min_pitch
                if not float(p).is_integer():
                    raise ValueError('Non integer pitches not yet supported.')
                p = int(p)
                if self.separate_instruments:
                    roll[t, p, i] = 1
                else:
                    roll[t, p, 0] = 0
        return roll

    def decode_to_midi(self, pianoroll):
        """Decodes pianoroll into midi."""
        # NOTE: Assumes four separate instruments ordered from high to low.
        midi_data = pretty_midi.PrettyMIDI()
        duration = self.qpm / 60 * self.shortest_duration
        tt, pp, ii = pianoroll.shape
        for i in range(ii):
            notes = []
            for p in range(pp):
                for t in range(tt):
                    if pianoroll[t, p, i]:
                        notes.append(
                            pretty_midi.Note(
                                velocity=100,
                                pitch=self.min_pitch + p,
                                start=t * duration,
                                end=(t + 1) * duration))
            notes = merge_held(notes)

            instrument = pretty_midi.Instrument(program=self.programs[i] - 1)
            instrument.notes.extend(notes)
            midi_data.instruments.append(instrument)
        return midi_data

    def encode_midi_melody_to_pianoroll(self, midi):
        """Encodes midi into pianorolls."""
        if len(midi.instruments) != 1:
            raise ValueError('Only one melody/instrument allowed, %r given.' %
                             (len(midi.instruments)))
        unused_tempo_change_times, tempo_changes = midi.get_tempo_changes()
        assert len(tempo_changes) == 1
        fs = 4
        # Returns matrix of shape (128, time) with summed velocities.
        roll = midi.get_piano_roll(fs=fs)  # 16th notes
        roll = np.where(roll > 0, 1, 0)
        logging.debug('Roll shape: %s', roll.shape)
        roll = roll.T
        logging.debug('Roll argmax: %s', np.argmax(roll, 1))
        return roll

    def encode_midi_to_pianoroll(self, midi, requested_shape):
        """Encodes midi into pianorolls according to requested_shape."""
        # TODO(annahuang): Generalize to not requiring a requested shape.
        # TODO(annahuang): Assign instruments to SATB according to range of notes.
        bb, tt, pp, ii = requested_shape
        if not midi.instruments:
            return np.zeros(requested_shape)
        elif len(midi.instruments) > ii:
            raise ValueError('Max number of instruments allowed %d < %d given.' % ii,
                             (len(midi.instruments)))
        unused_tempo_change_times, tempo_changes = midi.get_tempo_changes()
        assert len(tempo_changes) == 1

        logging.debug('# of instr %d', len(midi.instruments))
        # Encode each instrument separately.
        instr_rolls = [
            self.get_instr_pianoroll(instr, requested_shape)
            for instr in midi.instruments
        ]
        if len(instr_rolls) != ii:
            for unused_i in range(ii - len(instr_rolls)):
                instr_rolls.append(np.zeros_like(instr_rolls[0]))

        max_tt = np.max([roll.shape[0] for roll in instr_rolls])
        if tt < max_tt:
            logging.warning(
                'WARNING: input midi is a longer sequence then the requested'
                'size (%d > %d)', max_tt, tt)
        elif max_tt < tt:
            max_tt = tt
        pianorolls = np.zeros((bb, max_tt, pp, ii))
        for i, roll in enumerate(instr_rolls):
            pianorolls[:, :roll.shape[0], :, i] = np.tile(roll[:, :], (bb, 1, 1))
        logging.debug('Requested roll shape: %s', requested_shape)
        logging.debug('Roll argmax: %s',
                      np.argmax(pianorolls, axis=2) + self.min_pitch)
        return pianorolls

    def get_instr_pianoroll(self, midi_instr, requested_shape):
        """Returns midi_instr as 2D (time, model pitch_range) pianoroll."""
        pianoroll = np.zeros(requested_shape[1:-1])
        if not midi_instr.notes:
            return pianoroll
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(midi_instr)
        # TODO(annahuang): Sampling frequency is dataset dependent.
        fs = 4
        # Returns matrix of shape (128, time) with summed velocities.
        roll = midi.get_piano_roll(fs=fs)
        roll = np.where(roll > 0, 1, 0)
        roll = roll.T
        out_of_range_pitch_count = (
                np.sum(roll[:, self.max_pitch + 1:]) + np.sum(roll[:, :self.min_pitch]))
        if out_of_range_pitch_count > 0:
            raise ValueError(
                '%d pitches out of the range (%d, %d) the model was trained on.' %
                (out_of_range_pitch_count, self.min_pitch, self.max_pitch))
        roll = roll[:, self.min_pitch:self.max_pitch + 1]
        return roll


# https://github.com/magenta/magenta/blob/master/magenta/models/coconet/lib_pianoroll.py
def merge_held(notes):
    """Combine repeated notes into one sustained note."""
    notes = list(notes)
    i = 1
    while i < len(notes):
        if (notes[i].pitch == notes[i - 1].pitch and
                notes[i].start == notes[i - 1].end):
            notes[i - 1].end = notes[i].end
            del notes[i]
        else:
            i += 1
    return notes


def decode_to_midi(pianoroll, qpm=80, shortest_duration=1 / 16,
                   instrument='wind'):  # shortest_duration=1/16: 16th notes
    """Decodes pianoroll into midi."""
    # Note: the original code is expecting input of [t, num_pitch, output_channel],
    # changed to expect input of [t, output_channel].
    # pianoroll: [t, output_channel]
    # Oboe, English horn, clarinet, bassoon, sounds better on timidity.
    if instrument == 'wind':
        programs = [69, 70, 72, 71]
    elif instrument == 'piano':
        programs = [1, 1, 1, 1]
    # NOTE: Assumes four separate instruments ordered from high to low.
    midi_data = pretty_midi.PrettyMIDI()
    duration = qpm / 60 * shortest_duration
    tt, ii = pianoroll.shape
    for i in range(ii):
        notes = []
        for t in range(tt):
            notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=pianoroll[t, i],
                    start=t * duration,
                    end=(t + 1) * duration))
        notes = merge_held(notes)

        instrument = pretty_midi.Instrument(program=programs[i] - 1)
        instrument.notes.extend(notes)
        midi_data.instruments.append(instrument)
    return midi_data


def segment_pianoroll(data, window_length=48, hop_length=4, pad_value=0, nan_value=0):
    # data: list of [t, n_pitches, n_instruments]
    num_piece = len(data)
    data_all = []
    for i in range(num_piece):
        piece = data[i]
        if piece.shape[0] > window_length:
            data_list = [piece[i:i + window_length] for i in
                         range(0, piece.shape[0] - window_length + hop_length, hop_length)]
            if data_list[-1].shape[0] < window_length:
                data_list[-1] = piece[-window_length:]
            data_piece = np.stack(data_list, axis=0)
        else:
            data_piece = np.pad(piece, ((0, window_length - piece.shape[0]), (0, 0), (0, 0)), 'constant',
                                constant_values=pad_value)
            data_piece = data_piece[np.newaxis, ...]
        data_all.append(data_piece)
    return np.nan_to_num(np.concatenate(data_all, axis=0), nan=nan_value)


class BachPianorollNoteDataset(Dataset):

    def __init__(self, data, window_length=48, hop_length=4):
        self.window_length = window_length
        self.hop_length = hop_length
        pianoroll_encoder_decoder = PianorollEncoderDecoder(
            shortest_duration=0.125,
            min_pitch=34,  # changed to 34 for 48x48 grid
            max_pitch=81,
            separate_instruments=True,
            num_instruments=4,
            quantization_level=0.125
        )
        self.pianoroll = [pianoroll_encoder_decoder.encode(n) for n in data['note']]
        self.pianoroll = segment_pianoroll(self.pianoroll, window_length=window_length,
                                           hop_length=self.hop_length)
        self.note = [n.reshape(-1) for n in data['note']]
        self.note = segment_pianoroll(self.note, window_length=window_length * 4,
                                      hop_length=self.hop_length * 4)

    def __len__(self):
        return len(self.pianoroll)

    def __getitem__(self, item):
        single_batch_pianoroll = self.pianoroll[item]
        single_batch_pianoroll = single_batch_pianoroll.transpose((2, 0, 1))
        # single_batch_pianoroll: [n_instruments=4, t, n_pitches]
        single_batch_pianoroll = torch.tensor(single_batch_pianoroll)
        # single_batch_pianoroll_reduced: [1, t, n_pitches]
        single_batch_pianoroll_reduced = torch.clip(torch.sum(single_batch_pianoroll, 0, keepdim=True), min=0, max=1)

        return single_batch_pianoroll_reduced.float(), single_batch_pianoroll.long()


if __name__ == '__main__':
    bach_data_path = '../JsbChord16thSeparated.npy'
    from torch.utils.data.dataloader import DataLoader

    window_length = 48
    batch_size = 32
    num_workers = 4
    data = np.load(bach_data_path, allow_pickle=True, encoding='latin1')  # coconet

    training_data = BachPianorollNoteDataset(data.item()['train'], window_length=window_length)
    training_data = DataLoader(
        dataset=training_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        drop_last=True, collate_fn=None)
    evaluation_data = BachPianorollNoteDataset(data.item()['valid'], window_length=window_length)
    evaluation_data = DataLoader(
        dataset=evaluation_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        drop_last=False, collate_fn=None)

    data = next(iter(training_data))
    print(data[0].shape, data[1].shape)
