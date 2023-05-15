import json
import ndjson
from pathlib import Path

series = "24"
seasons = ["S%02d"%i for i in range(2,10)]
total_episodes = [24 for _ in range(7)] + [12]
assert len(seasons) == len(total_episodes)
save_dir_path = Path("/ssd_scratch/cvit/rodosingh/data/24/bassl/anno/")
videvents_path = Path("/ssd_scratch/cvit/rodosingh/data/24/")

#%% generate vid2idx.json data
vid2idx = {}
idx = 0
for i, season in enumerate(seasons):
    for j in range(total_episodes[i]):
        vid2idx[season + "E%02d"%(j+1)] = idx
        idx += 1
with open(save_dir_path/"vid2idx.json", "w") as f:
    json.dump(vid2idx, f)
print("vid2idx.json done!")

#%% generate ann.trainvaltest.ndjson and anno.inference.ndjson data
# {"video_id": "tt0047396", "shot_id": "0000", "num_shot": 792}
ann = []
for i, season in enumerate(seasons):
    for j in range(total_episodes[i]):
        with open(videvents_path/(f"{season}/{season}E{j+1:02d}/"+
                  f"videvents/{season}E{j+1:02d}.videvents"), "r") as f:
            num_shots = len(f.readlines())
        ann.extend([{"video_id": season + "E%02d"%(j+1), "shot_id": "%04d"%k, "num_shot": num_shots}
                    for k in range(num_shots)])
        print(f"{season}E{j+1:02d} done!")
        
file_name = ["anno.trainvaltest.ndjson", "anno.inference.ndjson"]
for fi in file_name:
    with open(save_dir_path/fi, "w") as f:
        ndjson.dump(ann, f)
    print(f"{fi} done!")
