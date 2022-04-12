from pathlib import Path
import json

from essentia.standard import TensorflowPredict, TensorTranspose
from essentia import Pool
import numpy as np

from feature_melspectrogram import (
    MelSpectrogramVGGish,
    MelSpectrogramMusiCNN,
    MelSpectrogramOpenL3,
)

FILE_PATH = Path(__file__).parent.absolute()


class EmbeddingModel:
    def __init__(
        self,
        model_type,
        model_path,
        hop_time=1,
        batch_size=60,
        fixed_batch_size=False,
        output="embeddings",
    ):
        self.model_type = model_type
        self.hop_time = hop_time
        self.batch_size = batch_size
        self.model_path = model_path
        self.output = output

        with open(FILE_PATH / "models_config.json", "r") as config_file:
            config = json.load(config_file)
        self.config = config[self.model_type]

        self.graph_path = Path(self.model_path)

        self.x_size = self.config["x_size"]
        self.y_size = self.config["y_size"]

        self.input_layer = self.config["input"]
        self.output_layer = self.config[self.output]

        self.seconds_to_patches = self.config["seconds_to_patches"]
        self.fixed_batch_size = fixed_batch_size

        if self.model_type in ("musicnn", "effnet_b0", "effnet_b0_3M"):
            self.mel_extractor = MelSpectrogramMusiCNN()
        elif self.model_type in ("vggish", "yamnet"):
            self.mel_extractor = MelSpectrogramVGGish()
        elif self.model_type == "openl3":
            self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time)

        if "batch_size" in self.config:
            if self.batch_size != self.config["batch_size"]:
                print(
                    f"changing batch size from ({self.batch_size}) to ({self.config['batch_size']}) as required by the model"
                )
            self.batch_size = self.config["batch_size"]
            self.fixed_batch_size = True

        self.model = TensorflowPredict(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer],
            squeeze=self.config["squeeze"],
        )

    def melspectrogram_to_activations(self, melspectrogram):
        # in OpenL3 the hop size is computed in the feature extraction level
        if self.model_type == "openl3":
            hop_size_samples = self.x_size
        else:
            hop_size_samples = int(self.hop_time * self.seconds_to_patches)

        batch = self.__melspectrogram_to_batch(melspectrogram, hop_size_samples)

        pool = Pool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)

            batch_len = end - start
            if (batch_len != self.batch_size) and self.fixed_batch_size:
                input_batch = np.zeros(
                    (self.batch_size, 1, self.x_size, self.y_size), dtype="float32"
                )
                input_batch[:batch_len] = batch[start:end]
            else:
                input_batch = batch[start:end]
            pool.set(self.input_layer, input_batch)
            out_pool = self.model(pool)
            output_batch = out_pool[self.output_layer].squeeze()

            if (batch_len != self.batch_size) and self.fixed_batch_size:
                output_batch = output_batch[:batch_len]

            embeddings.append(output_batch)

        return np.vstack(embeddings)

    def compute(self, audio_file):
        melspectrogram = self.mel_extractor.compute(audio_file)

        return self.melspectrogram_to_activations(melspectrogram)

    def compute_from_melspectrogram(self, melspectrogram_file):
        fp = np.memmap(melspectrogram_file, dtype="float16", mode="r+")
        melspectrogram = np.array(fp[:]).reshape((-1, self.y_size))

        return self.melspectrogram_to_activations(melspectrogram)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        if self.config["permutation"]:
            batch = TensorTranspose(permutation=self.config["permutation"])(batch)
        return batch
