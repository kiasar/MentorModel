import csv
import Corefunctions as Cf
from aitextgen.TokenDataset import TokenDataset
from aitextgen import aitextgen


def poem_data_maker(poems: list):
    return [" # ".join(song) + " # " for song in poems]


if __name__ == '__main__':
    PATH_TO_SONGS = "Data/songs_data.csv"
    print("Loading the songs file..........")
    with open(PATH_TO_SONGS, newline='') as f:
        songs = list(csv.reader(f))

    train_songs, val_songs, test_songs = Cf.data_cleaner(songs)
    all_songs = train_songs + val_songs + test_songs
    all_songs = poem_data_maker(all_songs)

    with open('data_for_gpt.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(all_songs))

    model = "EleutherAI/gpt-neo-125M"

    # ai = aitextgen(model=model, to_gpu=True)
    ai = aitextgen(tf_gpt2="774M", to_gpu=True)

    ai.train("data_for_gpt.txt",
             line_by_line=True,
             from_cache=False,
             padding_side='left',
             num_steps=100000,
             generate_every=500,
             save_every=500,
             save_gdrive=False,
             learning_rate=1e-3,
             fp16=False,
             batch_size=1,
             )

    ai.save()
    prompt_ai = aitextgen(model_folder='.', to_gpu=True)
    print(prompt_ai.generate())
