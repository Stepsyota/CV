CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

LABEL_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}

INPUT_DIM = 32 * 32 * 3
NUM_CLASSES = 10
