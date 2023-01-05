# automatic_music_transcription

Automatic transcription of monophonic instrumental audio to music sheet format.

Printed Images of Music Staves (PrIMuS) dataset containing 87678 monophonic real-music incipits was used for this project. For each incipit, the dataset contains a MIDI file along with a semantic representation.

Example of semantic representation:

```clef-C1	keySignature-EbM	timeSignature-2/4	multirest-23	barline	rest-quarter	rest-eighth	note-Bb4_eighth	barline	note-Bb4_quarter.	note-G4_eighth	barline	note-Eb5_quarter.	note-D5_eighth	barline	note-C5_eighth	note-C5_eighth	rest-quarter	barline```

7 instrument soundfonts were used to convert the MIDI files into .wav format of said instruments. The instruments that were tested were: acoustic guitar, electric bass, marimba, piano, recorder, triangle, and violin.

These .wav files of each instruments were then converted to Mel spectrograms using `mel.py` and `process.py`. These spectrogram images of varying width were then sorted and padded into batches for model training using `batch.py`. Every music note and symbol in the semantic files was tokenized by `tokenizer.py` and the mappings are saved in `model_data` in .json format. Any music symbol that cannot be recognized from the audio (barline, keySignature, clef, timeSignature) was removed. The tokenized representation is stored in .npy format, and is also batched as training target with `batch.py`.

The CRNN model used was adapted from https://www.ijitee.org/wp-content/uploads/papers/v9i8/H6264069820.pdf and it was used with CTC loss. Extra convolutional and pooling layers were added because our spectrogram images have greater height than the OCR images used in the paper. The pooling layers were also adjusted to only pool across the vertical axis, and not the horizontal axis. This is because our horizontal axis represents time and therefore cannot be pooled (vertical axis is frequency). The input size has been set to (128, None) since each batch of images have different widths but the height will always be 128 pixels (our spectrograms have 128 Mel bins). The shape of the label tensor was set to the highest number of tokens in any semantic file in the dataset, which is 51. The output dense layer's size has been set to the size of our token vocabulary to be able to predict every token.

Model training and testing was done in the `model.ipynb` Jupyter Notebook. The model can also be trained non-interactively by using `train.py`. The metric used to evaluate the transcription result is the Character Error Rate (CER). This can be calculated as the Levenshtein distance between the original and predicted transcription (in text form, not tokenized), divided by the length of the original transcription. The lower the CER, the better and 0 CER represents a perfect score. Optionally accuracy can be obtained with 1 - CER.

The instrument transcription results are as follows (CER): 
1. Piano: 0.0546
2. Acoustic Guitar: 0.0627
3. Marimba: 0.0640
4. Triangle: 0.0952
5. Violin: 0.1059
6. Recorder: 0.1131
7. Electric Bass: 0.3826

The next step for this project is to train the models again on .kern format as the training target and desired output (instead of semantic format). Open source code has already been written to convert .kern format to music sheet format: https://verovio.humdrum.org/. The end goal for this project is to make an end-to-end audio to music sheet transcription.

Finally, a PowerPoint presentation on this project has also been added to the repository.