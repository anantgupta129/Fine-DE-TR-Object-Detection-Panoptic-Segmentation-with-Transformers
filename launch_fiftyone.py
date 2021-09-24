import fiftyone as fo
import fiftyone.zoo as foz

dataset = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["panoptic"],
)

dataset = foz.load_zoo_dataset(dataset)
session = fo.launch_app(dataset)

session.wait()
