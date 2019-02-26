## Installation

```
pip3 install -r requirements.txt
```

To verify that the packages are installed successfully, run the following command to see a round of Pong gameplay rendered on screen:

```
python3 -m trainer.task --render
```

This package comes with a trained checkpoint.  To play a game with it, run:
```
python3 -m trainer.task --render --restore --output-dir ./demo-checkpoint
```

You can also run the following command to train locally.
By default, outputs (checkpoints and TensorBoard summaries) are written to `/tmp/hashcomp_output`:

```
python3 -m trainer.task
```

## Run training job

To submit the training job to Cloud Machine Learning Engine:

```
in submit.sh
    modify BUCKET="gs://YOUR-BUCKET/"
bash submit.sh
```