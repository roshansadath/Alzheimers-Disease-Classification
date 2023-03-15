from alzheimerdetection.data import load_alzheimer_mri_dataset, load_alzheimer_mri_dataset_test, \
    load_alzheimer_mri_dataset_train


def test_load_train():
    dataset = load_alzheimer_mri_dataset_train()
    assert len(dataset) == 5121


def test_load_test():
    dataset = load_alzheimer_mri_dataset_test()
    assert len(dataset) == 1279


def test_load_all():
    train, test = load_alzheimer_mri_dataset()
    assert len(train) == 5121
    assert len(test) == 1279


def test_data_shape():
    dataset = load_alzheimer_mri_dataset_train()
    instance = dataset[0]

    assert dataset.classes == ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    assert instance[0].size == (176, 208)
    assert instance[1] == 0
