from cvutils.transform.utils import size_cmp


def test_size_cmp():
    assert size_cmp(2, 1) == -1
    assert size_cmp(1, 2) == 1
    assert size_cmp(1, 1) == 0

    assert size_cmp([3, 224, 224], [3, 224, 224]) == 0
    assert size_cmp((3, 224, 224), (3, 224, 224)) == 0

    assert size_cmp([3, 225, 224], [3, 224, 224]) == -1
    assert size_cmp((3, 225, 224), (3, 224, 224)) == -1

    assert size_cmp([3, 224, 225], [3, 224, 224]) == -1
    assert size_cmp((3, 224, 225), (3, 224, 224)) == -1

    assert size_cmp([3, 222, 224], [3, 224, 224]) == 1
    assert size_cmp((3, 222, 224), (3, 224, 224)) == 1

    assert size_cmp([3, 224, 222], [3, 224, 224]) == 1
    assert size_cmp((3, 224, 222), (3, 224, 224)) == 1

    assert size_cmp([3, 222, 225], [3, 224, 224]) == 1
    assert size_cmp((3, 222, 225), (3, 224, 224)) == 1
