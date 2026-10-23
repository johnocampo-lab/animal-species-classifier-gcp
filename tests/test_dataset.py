from data.dataset import get_dataloaders, get_transforms


def test_transforms_differ_train_vs_eval():
    train_tf, eval_tf = get_transforms(224)
    assert train_tf is not eval_tf
    assert len(train_tf.transforms) > len(eval_tf.transforms)


def test_imagefolder_loaders(demo_imagefolder):
    loaders, classes = get_dataloaders(str(demo_imagefolder), batch_size=4, num_workers=0)
    assert classes == ["bird", "cat", "deer", "dog", "frog", "horse"]
    batch = next(iter(loaders["train"]))
    assert batch[0].shape[0] <= 4
    assert batch[1].ndim == 1
