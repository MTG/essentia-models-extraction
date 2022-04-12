# Essentia Models extraction
Batch extractor for melspectrograms, embeddings, and activations for the Essentia models.


## Install
The dependecies can be installed via `pip`:

```
python3 -m venv venv

source venv/bin/activate

pip install pip --upgrade

pip install -r requirements.txt
```

### GPU support
The extraction of activations and embeddings can be speeded up if a GPU is available.
`essentia-tensorflow` currently requires `cuDNN=8.1` and `CUDA=11.2` for this.

It is the responsibility of the user to install such dependencies.
In case [conda](https://docs.conda.io) is available, the libraries can be easily installed:

```
conda install -c conda-forge cudnn=8.1
conda install -c conda-forge cudatoolkit=11.2
```


## Usage
Check the [examples](examples/) for the typical use cases.

The extraction of embeddings or activations requires a model.
Check the [Essentia Models](https://essentia.upf.edu/models.html) site to download the appropriate model.

These are all the options suported by the script:

```
usage: batch extractor for melspectrograms, embeddings, and activations for the Essentia models
       [-h]
       [--feature {musicnn_melspectrogram,vggish_melspectrogram,openl3_melspectrogram,effnet_b0,effnet_b0_3M,musicnn,openl3,vggish,yamnet}]
       [--model-path MODEL_PATH] [--from-melspectrogram] [--force] [--dry-run]
       [--output {embeddings,activations}] [--batch-size BATCH_SIZE]
       [--hop-time HOP_TIME]
       input_folder output_folder filelist

positional arguments:
  input_folder          base folder with the audio
  output_folder         base folder for the output features
  filelist              path to files from `input_folder`

optional arguments:
  -h, --help            show this help message and exit
  --feature {musicnn_melspectrogram,vggish_melspectrogram,openl3_melspectrogram,effnet_b0,effnet_b0_3M,musicnn,openl3,vggish,yamnet}
                        the feature to extract. Can be melspectrogram or
                        embeddings
  --model-path MODEL_PATH
                        path to the model file (only required for embeddings
                        and activations)
  --from-melspectrogram
                        extract the embeddings or activations from pre-
                        computed melspectrograms (speed-up the process)
  --force               recompute if the targets already exists
  --dry-run             list files to process without computing
  --output {embeddings,activations}
                        whether to extract embeddings or activatons
  --batch-size BATCH_SIZE
                        batch size for the analysis (only noticeable when a
                        GPU is available)
  --hop-time HOP_TIME   hop time for the embeddings or activations in seconds.
```

## Tips
The feature extraction can be parallelized to several processes, possibly with dedicated GPU for each one.

To split a `filelist` into `n` chunks:
```
split -n l/`n` filelist filelist.
```
