def test_ImageLoader_singlefile():
    import pyWitnessAI

    il = pyWitnessAI.ImageLoader("./data/01_Georgia_State_Video1/Video1_Mugshot2.png")

    assert len(il.images) == 1


def test_ImageLoader_directory():
    import pyWitnessAI

    il = pyWitnessAI.ImageLoader("./data/01_Georgia_State_Video1/")

    assert len(il.images) == 8


def test_ImageLoader_wildcard():
    import pyWitnessAI

    il = pyWitnessAI.ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    assert len(il.images) == 6

def test_ImageLoader_list():
    import pyWitnessAI

    il = pyWitnessAI.ImageLoader(["./data/01_Georgia_State_Video1/*Mugshot*",
                                  "./data/01_Georgia_State_Video1/Video1_ProbeImage.png"])

    assert len(il.images) == 7
