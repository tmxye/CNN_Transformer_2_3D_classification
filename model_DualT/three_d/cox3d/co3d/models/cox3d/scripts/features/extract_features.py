import argparse
import pickle
from pathlib import Path

import torch
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
)
from tqdm import tqdm

from datasets.thumos14 import decode_video
from datasets.transforms import RandomShortSideScaleJitterVideo
from models.cox3d.modules.x3d import CoX3D

anno = {
    "train_session_set": [
        "video_validation_0000690",
        "video_validation_0000288",
        "video_validation_0000289",
        "video_validation_0000416",
        "video_validation_0000282",
        "video_validation_0000283",
        "video_validation_0000281",
        "video_validation_0000286",
        "video_validation_0000287",
        "video_validation_0000284",
        "video_validation_0000285",
        "video_validation_0000202",
        "video_validation_0000203",
        "video_validation_0000201",
        "video_validation_0000206",
        "video_validation_0000207",
        "video_validation_0000204",
        "video_validation_0000205",
        "video_validation_0000790",
        "video_validation_0000208",
        "video_validation_0000209",
        "video_validation_0000420",
        "video_validation_0000364",
        "video_validation_0000853",
        "video_validation_0000950",
        "video_validation_0000937",
        "video_validation_0000367",
        "video_validation_0000290",
        "video_validation_0000210",
        "video_validation_0000059",
        "video_validation_0000058",
        "video_validation_0000057",
        "video_validation_0000056",
        "video_validation_0000055",
        "video_validation_0000054",
        "video_validation_0000053",
        "video_validation_0000052",
        "video_validation_0000051",
        "video_validation_0000933",
        "video_validation_0000949",
        "video_validation_0000948",
        "video_validation_0000945",
        "video_validation_0000944",
        "video_validation_0000947",
        "video_validation_0000946",
        "video_validation_0000941",
        "video_validation_0000940",
        "video_validation_0000190",
        "video_validation_0000942",
        "video_validation_0000261",
        "video_validation_0000262",
        "video_validation_0000263",
        "video_validation_0000264",
        "video_validation_0000265",
        "video_validation_0000266",
        "video_validation_0000267",
        "video_validation_0000268",
        "video_validation_0000269",
        "video_validation_0000989",
        "video_validation_0000060",
        "video_validation_0000370",
        "video_validation_0000938",
        "video_validation_0000935",
        "video_validation_0000668",
        "video_validation_0000669",
        "video_validation_0000664",
        "video_validation_0000665",
        "video_validation_0000932",
        "video_validation_0000667",
        "video_validation_0000934",
        "video_validation_0000661",
        "video_validation_0000662",
        "video_validation_0000663",
        "video_validation_0000181",
        "video_validation_0000180",
        "video_validation_0000183",
        "video_validation_0000182",
        "video_validation_0000185",
        "video_validation_0000184",
        "video_validation_0000187",
        "video_validation_0000186",
        "video_validation_0000189",
        "video_validation_0000188",
        "video_validation_0000936",
        "video_validation_0000270",
        "video_validation_0000854",
        "video_validation_0000178",
        "video_validation_0000179",
        "video_validation_0000174",
        "video_validation_0000175",
        "video_validation_0000176",
        "video_validation_0000177",
        "video_validation_0000170",
        "video_validation_0000171",
        "video_validation_0000172",
        "video_validation_0000173",
        "video_validation_0000670",
        "video_validation_0000419",
        "video_validation_0000943",
        "video_validation_0000485",
        "video_validation_0000369",
        "video_validation_0000368",
        "video_validation_0000318",
        "video_validation_0000319",
        "video_validation_0000415",
        "video_validation_0000414",
        "video_validation_0000413",
        "video_validation_0000412",
        "video_validation_0000411",
        "video_validation_0000311",
        "video_validation_0000312",
        "video_validation_0000313",
        "video_validation_0000314",
        "video_validation_0000315",
        "video_validation_0000316",
        "video_validation_0000317",
        "video_validation_0000418",
        "video_validation_0000365",
        "video_validation_0000482",
        "video_validation_0000169",
        "video_validation_0000168",
        "video_validation_0000167",
        "video_validation_0000166",
        "video_validation_0000165",
        "video_validation_0000164",
        "video_validation_0000163",
        "video_validation_0000162",
        "video_validation_0000161",
        "video_validation_0000160",
        "video_validation_0000857",
        "video_validation_0000856",
        "video_validation_0000855",
        "video_validation_0000366",
        "video_validation_0000488",
        "video_validation_0000489",
        "video_validation_0000851",
        "video_validation_0000484",
        "video_validation_0000361",
        "video_validation_0000486",
        "video_validation_0000487",
        "video_validation_0000481",
        "video_validation_0000910",
        "video_validation_0000483",
        "video_validation_0000363",
        "video_validation_0000990",
        "video_validation_0000939",
        "video_validation_0000362",
        "video_validation_0000987",
        "video_validation_0000859",
        "video_validation_0000787",
        "video_validation_0000786",
        "video_validation_0000785",
        "video_validation_0000784",
        "video_validation_0000783",
        "video_validation_0000782",
        "video_validation_0000781",
        "video_validation_0000981",
        "video_validation_0000983",
        "video_validation_0000982",
        "video_validation_0000985",
        "video_validation_0000984",
        "video_validation_0000417",
        "video_validation_0000788",
        "video_validation_0000152",
        "video_validation_0000153",
        "video_validation_0000151",
        "video_validation_0000156",
        "video_validation_0000157",
        "video_validation_0000154",
        "video_validation_0000155",
        "video_validation_0000158",
        "video_validation_0000159",
        "video_validation_0000901",
        "video_validation_0000903",
        "video_validation_0000902",
        "video_validation_0000905",
        "video_validation_0000904",
        "video_validation_0000907",
        "video_validation_0000906",
        "video_validation_0000909",
        "video_validation_0000908",
        "video_validation_0000490",
        "video_validation_0000860",
        "video_validation_0000858",
        "video_validation_0000988",
        "video_validation_0000320",
        "video_validation_0000688",
        "video_validation_0000689",
        "video_validation_0000686",
        "video_validation_0000687",
        "video_validation_0000684",
        "video_validation_0000685",
        "video_validation_0000682",
        "video_validation_0000683",
        "video_validation_0000681",
        "video_validation_0000789",
        "video_validation_0000986",
        "video_validation_0000931",
        "video_validation_0000852",
        "video_validation_0000666",
    ],
    "test_session_set": [
        "video_test_0000292",
        "video_test_0001078",
        "video_test_0000896",
        "video_test_0000897",
        "video_test_0000950",
        "video_test_0001159",
        "video_test_0001079",
        "video_test_0000807",
        "video_test_0000179",
        "video_test_0000173",
        "video_test_0001072",
        "video_test_0001075",
        "video_test_0000767",
        "video_test_0001076",
        "video_test_0000007",
        "video_test_0000006",
        "video_test_0000556",
        "video_test_0001307",
        "video_test_0001153",
        "video_test_0000718",
        "video_test_0000716",
        "video_test_0001309",
        "video_test_0000714",
        "video_test_0000558",
        "video_test_0001267",
        "video_test_0000367",
        "video_test_0001324",
        "video_test_0000085",
        "video_test_0000887",
        "video_test_0001281",
        "video_test_0000882",
        "video_test_0000671",
        "video_test_0000964",
        "video_test_0001164",
        "video_test_0001114",
        "video_test_0000771",
        "video_test_0001163",
        "video_test_0001118",
        "video_test_0001201",
        "video_test_0001040",
        "video_test_0001207",
        "video_test_0000723",
        "video_test_0000569",
        "video_test_0000672",
        "video_test_0000673",
        "video_test_0000278",
        "video_test_0001162",
        "video_test_0000405",
        "video_test_0000073",
        "video_test_0000560",
        "video_test_0001276",
        "video_test_0000270",
        "video_test_0000273",
        "video_test_0000374",
        "video_test_0000372",
        "video_test_0001168",
        "video_test_0000379",
        "video_test_0001446",
        "video_test_0001447",
        "video_test_0001098",
        "video_test_0000873",
        "video_test_0000039",
        "video_test_0000442",
        "video_test_0001219",
        "video_test_0000762",
        "video_test_0000611",
        "video_test_0000617",
        "video_test_0000615",
        "video_test_0001270",
        "video_test_0000740",
        "video_test_0000293",
        "video_test_0000504",
        "video_test_0000505",
        "video_test_0000665",
        "video_test_0000664",
        "video_test_0000577",
        "video_test_0000814",
        "video_test_0001369",
        "video_test_0001194",
        "video_test_0001195",
        "video_test_0001512",
        "video_test_0001235",
        "video_test_0001459",
        "video_test_0000691",
        "video_test_0000765",
        "video_test_0001452",
        "video_test_0000188",
        "video_test_0000591",
        "video_test_0001268",
        "video_test_0000593",
        "video_test_0000864",
        "video_test_0000601",
        "video_test_0001135",
        "video_test_0000004",
        "video_test_0000903",
        "video_test_0000285",
        "video_test_0001174",
        "video_test_0000046",
        "video_test_0000045",
        "video_test_0001223",
        "video_test_0001358",
        "video_test_0001134",
        "video_test_0000698",
        "video_test_0000461",
        "video_test_0001182",
        "video_test_0000450",
        "video_test_0000602",
        "video_test_0001229",
        "video_test_0000989",
        "video_test_0000357",
        "video_test_0001039",
        "video_test_0000355",
        "video_test_0000353",
        "video_test_0001508",
        "video_test_0000981",
        "video_test_0000242",
        "video_test_0000854",
        "video_test_0001484",
        "video_test_0000635",
        "video_test_0001129",
        "video_test_0001339",
        "video_test_0001483",
        "video_test_0001123",
        "video_test_0001127",
        "video_test_0000689",
        "video_test_0000756",
        "video_test_0001431",
        "video_test_0000129",
        "video_test_0001433",
        "video_test_0001343",
        "video_test_0000324",
        "video_test_0001064",
        "video_test_0001531",
        "video_test_0001532",
        "video_test_0000413",
        "video_test_0000991",
        "video_test_0001255",
        "video_test_0000464",
        "video_test_0001202",
        "video_test_0001080",
        "video_test_0001081",
        "video_test_0000847",
        "video_test_0000028",
        "video_test_0000844",
        "video_test_0000622",
        "video_test_0000026",
        "video_test_0001325",
        "video_test_0001496",
        "video_test_0001495",
        "video_test_0000624",
        "video_test_0000724",
        "video_test_0001409",
        "video_test_0000131",
        "video_test_0000448",
        "video_test_0000444",
        "video_test_0000443",
        "video_test_0001038",
        "video_test_0000238",
        "video_test_0001527",
        "video_test_0001522",
        "video_test_0000051",
        "video_test_0001058",
        "video_test_0001391",
        "video_test_0000429",
        "video_test_0000426",
        "video_test_0000785",
        "video_test_0000786",
        "video_test_0001314",
        "video_test_0000392",
        "video_test_0000423",
        "video_test_0001146",
        "video_test_0001313",
        "video_test_0001008",
        "video_test_0001247",
        "video_test_0000737",
        "video_test_0001319",
        "video_test_0000308",
        "video_test_0000730",
        "video_test_0000058",
        "video_test_0000538",
        "video_test_0001556",
        "video_test_0000113",
        "video_test_0000626",
        "video_test_0000839",
        "video_test_0000220",
        "video_test_0001389",
        "video_test_0000437",
        "video_test_0000940",
        "video_test_0000211",
        "video_test_0000946",
        "video_test_0001558",
        "video_test_0000796",
        "video_test_0000062",
        "video_test_0000793",
        "video_test_0000987",
        "video_test_0001066",
        "video_test_0000412",
        "video_test_0000798",
        "video_test_0001549",
        "video_test_0000011",
        "video_test_0001257",
        "video_test_0000541",
        "video_test_0000701",
        "video_test_0000250",
        "video_test_0000254",
        "video_test_0000549",
        "video_test_0001209",
        "video_test_0001463",
        "video_test_0001460",
        "video_test_0000319",
        "video_test_0001468",
        "video_test_0000846",
        "video_test_0001292",
    ],
}


def main(model, split):
    num_classes = 22
    dim_in = 3
    temporal_window_size = 1

    # Select hparams
    if model == "s":
        image_size = 160
        x3d_conv1_dim = 12
        x3d_conv5_dim = 2048
        x3d_num_groups = 1
        x3d_width_per_group = 64
        x3d_width_factor = 2.0
        x3d_depth_factor = 2.2
        x3d_bottleneck_factor = 2.25
        x3d_use_channelwise_3x3x3 = 1
        x3d_dropout_rate = 0.5
        x3d_head_activation = "softmax"
        x3d_head_batchnorm = 0
        x3d_fc_std_init = 0.01
        x3d_final_batchnorm_zero_init = 1
        target_fps = 5.0
    elif model == "m":
        image_size = 224
        x3d_num_groups = 1
        x3d_width_per_group = 64
        x3d_width_factor = 2.0
        x3d_depth_factor = 2.2
        x3d_bottleneck_factor = 2.25
        x3d_conv1_dim = 12
        x3d_conv5_dim = 2048
        x3d_use_channelwise_3x3x3 = 1
        x3d_dropout_rate = 0.5
        x3d_head_activation = "softmax"
        x3d_head_batchnorm = 0
        x3d_fc_std_init = 0.01
        x3d_final_batchnorm_zero_init = 1
        target_fps = 6.0
    elif model == "l":
        image_size = 312
        x3d_num_groups = 1
        x3d_width_per_group = 64
        x3d_width_factor = 2.0
        x3d_depth_factor = 5.0
        x3d_bottleneck_factor = 2.25
        x3d_conv1_dim = 12
        x3d_conv5_dim = 2048
        x3d_use_channelwise_3x3x3 = 1
        x3d_dropout_rate = 0.5
        x3d_head_activation = "softmax"
        x3d_head_batchnorm = 0
        x3d_fc_std_init = 0.01
        x3d_final_batchnorm_zero_init = 1
        target_fps = 6.0

    # Define model
    module = CoX3D(
        dim_in,
        image_size,
        temporal_window_size,
        num_classes,
        x3d_conv1_dim,
        x3d_conv5_dim,
        x3d_num_groups,
        x3d_width_per_group,
        x3d_width_factor,
        x3d_depth_factor,
        x3d_bottleneck_factor,
        x3d_use_channelwise_3x3x3,
        x3d_dropout_rate,
        x3d_head_activation,
        x3d_head_batchnorm,
        x3d_fc_std_init,
        x3d_final_batchnorm_zero_init,
        temporal_fill="zeros",
        se_scope="frame",
        headless=True,
    )
    module.call_mode = "forward_steps"

    # Load weights
    new_model_state = module.state_dict(flatten=True)
    state_dict = torch.load(
        f"/home/lh/projects/co3d/models/x3d/weights/x3d_{model}.pyth",
        map_location="cpu",
    )["model_state"]

    def key_ok(k):
        return k in new_model_state and k in state_dict

    def size_ok(k):
        return new_model_state[k].size() == state_dict[k].size()

    to_load = {k: v for k, v in state_dict.items() if key_ok(k) and size_ok(k)}

    module.load_state_dict(to_load, flatten=True, strict=False)

    # Prepare transforms

    MEAN, STD = ((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
    # train_crop_pix, train_scale_pix_min, train_scale_pix_max = (140.0, 160, 200.0)
    # train_transforms = Compose(
    #     [
    #         ToTensorVideo(),
    #         RandomShortSideScaleJitterVideo(
    #             min_size=train_scale_pix_min, max_size=train_scale_pix_max
    #         ),
    #         CenterCropVideo(image_size),
    #         NormalizeVideo(mean=MEAN, std=STD),
    #     ]
    # )

    eval_transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(min_size=image_size, max_size=image_size),
            CenterCropVideo(image_size),
            NormalizeVideo(mean=MEAN, std=STD),
        ]
    )

    # Identify videos
    ds_path = Path(f"/mnt/archive/common/datasets/thumos14/data/{split}")
    vid_paths = list(ds_path.glob("*.mp4"))

    vid_paths = [
        Path(f"/mnt/archive/common/datasets/thumos14/data/{split}/{vid}.mp4")
        for vid in anno[f"{'train' if split == 'val' else 'test'}_session_set"]
    ]

    # Extract features
    device = torch.device("cuda")
    module = module.to(device=device)
    module.eval()
    block_size = 256

    features = {}

    print(f"Extracting CoX3D-{model} features for THUMOS14-{split}")
    with torch.no_grad():
        # Iterate through dataset, extracting features meanwhile
        for p in tqdm(vid_paths):
            video = decode_video(str(p), 0, -1, target_fps)

            if video is None:
                print(f"Error decoding {p}")
                continue

            video = video
            video = eval_transforms(video).unsqueeze(0)

            # Warm up module
            module.clean_state()
            # module.warm_up(video.shape[:2] + video.shape[3:])

            T = video.shape[2]

            # Save in blocks
            feat = []
            for start in range(0, T, block_size):
                end = min(T, start + block_size)

                f = module.forward_steps(video[:, :, start:end].to(device=device))
                if len(f.shape) > 3:
                    f = f.mean(dim=(-1, -2))
                if isinstance(f, torch.Tensor):
                    feat.append(f.detach().to(device=torch.device("cpu")))

            if len(feat) > 0:
                features[p.stem] = torch.cat(feat, dim=2)

    # Save features
    with open(f"cox3d_{split}_kin_features_{split}.pickle", "wb") as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="s", choices=["s", "m", "l"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    main(args.model, args.split)
