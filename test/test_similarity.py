from pyWitnessAI import ImageLoader, ImageAnalyzer

# ---------- Georgia State ----------
def test_similarity_georgia_pipeline():
    """
    Test to reproduce the similarity scores from Kleider-Offutt et al. (2024)
    using the dedicated process_georgia_pipeline method.
    """
    # Load images as specified in the notebook's code cell for the Georgia pipeline
    WH_column_images_Georgia = ImageLoader("./data/01_Georgia_State_Video1/Video1_ProbeImage.png")
    WH_row_images_Georgia = ImageLoader([
        "./data/01_Georgia_State_Video1/*Mugshot*",
        "./data/01_Georgia_State_Video1/Video1_Perpetrator.png"
    ])

    image_analyzer_Georgia = ImageAnalyzer(
        column_images=WH_column_images_Georgia,
        row_images=WH_row_images_Georgia,
    )

    image_analyzer_Georgia.process_georgia_pipeline()
    df = image_analyzer_Georgia.dataframe()

    # Expected values updated from the notebook
    assert df.loc["Video1_Perpetrator", "Video1_ProbeImage"] == 0.8759
    assert df.loc["Video1_Mugshot2", "Video1_ProbeImage"] == 1.2342
    assert df.loc["Video1_Mugshot3", "Video1_ProbeImage"] == 1.3474
    assert df.loc["Video1_Mugshot4", "Video1_ProbeImage"] == 1.2398
    assert df.loc["Video1_Mugshot5", "Video1_ProbeImage"] == 1.2717
    assert df.loc["Video1_Mugshot6", "Video1_ProbeImage"] == 1.2998
    assert df.loc["Video1_Mugshot7", "Video1_ProbeImage"] == 1.2906


# ---------- different distance metrics ----------
def test_similarity_mtcnn_facenet_euclidean_l2():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        align=False,
        enforce_detection=False,
        model="Facenet",
        backend="mtcnn",
        distance_metric="euclidean_l2",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 1.0869
    assert df["Video1_Perpetrator"][1] == 1.2359
    assert df["Video1_Perpetrator"][2] == 1.2768
    assert df["Video1_Perpetrator"][3] == 1.3133
    assert df["Video1_Perpetrator"][4] == 1.1511
    assert df["Video1_Perpetrator"][5] == 1.2442


def test_similarity_mtcnn_facenet_cosine():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="mtcnn",
        distance_metric="cosine",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 0.5906
    assert df["Video1_Perpetrator"][1] == 0.7637
    assert df["Video1_Perpetrator"][2] == 0.8151
    assert df["Video1_Perpetrator"][3] == 0.8624
    assert df["Video1_Perpetrator"][4] == 0.6626
    assert df["Video1_Perpetrator"][5] == 0.7740


def test_similarity_mtcnn_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="mtcnn",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 13.0366
    assert df["Video1_Perpetrator"][1] == 14.7332
    assert df["Video1_Perpetrator"][2] == 15.2573
    assert df["Video1_Perpetrator"][3] == 15.6377
    assert df["Video1_Perpetrator"][4] == 13.4157
    assert df["Video1_Perpetrator"][5] == 14.8153


# ---------- different detection backends ----------
def test_similarity_opencv_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="opencv",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 11.6449
    assert df["Video1_Perpetrator"][1] == 14.3952
    assert df["Video1_Perpetrator"][2] == 14.1869
    assert df["Video1_Perpetrator"][3] == 14.9362
    assert df["Video1_Perpetrator"][4] == 13.5124
    assert df["Video1_Perpetrator"][5] == 14.2616


def test_similarity_fastmtcnn_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="fastmtcnn",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 13.0325
    assert df["Video1_Perpetrator"][1] == 14.7854
    assert df["Video1_Perpetrator"][2] == 15.0674
    assert df["Video1_Perpetrator"][3] == 15.9633
    assert df["Video1_Perpetrator"][4] == 13.4093
    assert df["Video1_Perpetrator"][5] == 14.8043


def test_similarity_ssd_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="ssd",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 12.7515
    assert df["Video1_Perpetrator"][1] == 15.0852
    assert df["Video1_Perpetrator"][2] == 14.4758
    assert df["Video1_Perpetrator"][3] == 15.2980
    assert df["Video1_Perpetrator"][4] == 13.4211
    assert df["Video1_Perpetrator"][5] == 14.8717


def test_similarity_dlib_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="dlib",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 12.1419
    assert df["Video1_Perpetrator"][1] == 15.4491
    assert df["Video1_Perpetrator"][2] == 14.0794
    assert df["Video1_Perpetrator"][3] == 14.8101
    assert df["Video1_Perpetrator"][4] == 13.8467
    assert df["Video1_Perpetrator"][5] == 15.0814


def test_similarity_retinaface_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="retinaface",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 14.0947
    assert df["Video1_Perpetrator"][1] == 15.2000
    assert df["Video1_Perpetrator"][2] == 14.5374
    assert df["Video1_Perpetrator"][3] == 14.6287
    assert df["Video1_Perpetrator"][4] == 12.7990
    assert df["Video1_Perpetrator"][5] == 14.7612


def test_similarity_yunet_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="yunet",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 12.5380
    assert df["Video1_Perpetrator"][1] == 15.0752
    assert df["Video1_Perpetrator"][2] == 14.7540
    assert df["Video1_Perpetrator"][3] == 15.5789
    assert df["Video1_Perpetrator"][4] == 13.1358
    assert df["Video1_Perpetrator"][5] == 15.3376


def test_similarity_centerface_facenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="centerface",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 12.6939
    assert df["Video1_Perpetrator"][1] == 15.1955
    assert df["Video1_Perpetrator"][2] == 14.4030
    assert df["Video1_Perpetrator"][3] == 15.2244
    assert df["Video1_Perpetrator"][4] == 13.4164
    assert df["Video1_Perpetrator"][5] == 15.1770


# ---------- different models ----------
def test_similarity_mtcnn_vggface_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="VGG-Face",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 1.2198
    assert df["Video1_Perpetrator"][1] == 1.3102
    assert df["Video1_Perpetrator"][2] == 1.2750
    assert df["Video1_Perpetrator"][3] == 1.2426
    assert df["Video1_Perpetrator"][4] == 1.2195
    assert df["Video1_Perpetrator"][5] == 1.2934


def test_similarity_mtcnn_facenet512_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet512",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 24.0495
    assert df["Video1_Perpetrator"][1] == 22.9667
    assert df["Video1_Perpetrator"][2] == 25.2377
    assert df["Video1_Perpetrator"][3] == 26.8565
    assert df["Video1_Perpetrator"][4] == 23.8400
    assert df["Video1_Perpetrator"][5] == 28.4808


def test_similarity_mtcnn_openface_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="OpenFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 0.8041
    assert df["Video1_Perpetrator"][1] == 0.9918
    assert df["Video1_Perpetrator"][2] == 0.9186
    assert df["Video1_Perpetrator"][3] == 0.8266
    assert df["Video1_Perpetrator"][4] == 0.8583
    assert df["Video1_Perpetrator"][5] == 0.8997


def test_similarity_mtcnn_deepid_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="DeepID",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 53.9863
    assert df["Video1_Perpetrator"][1] == 62.8030
    assert df["Video1_Perpetrator"][2] == 64.1543
    assert df["Video1_Perpetrator"][3] == 58.0046
    assert df["Video1_Perpetrator"][4] == 93.2651
    assert df["Video1_Perpetrator"][5] == 55.7571


def test_similarity_mtcnn_arcface_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="ArcFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 5.5961
    assert df["Video1_Perpetrator"][1] == 5.3795
    assert df["Video1_Perpetrator"][2] == 5.7088
    assert df["Video1_Perpetrator"][3] == 5.9477
    assert df["Video1_Perpetrator"][4] == 5.2541
    assert df["Video1_Perpetrator"][5] == 5.9396


def test_similarity_mtcnn_dlib_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Dlib",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 0.6366
    assert df["Video1_Perpetrator"][1] == 0.6068
    assert df["Video1_Perpetrator"][2] == 0.6362
    assert df["Video1_Perpetrator"][3] == 0.6377
    assert df["Video1_Perpetrator"][4] == 0.6038
    assert df["Video1_Perpetrator"][5] == 0.6575


def test_similarity_mtcnn_sface_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="SFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 11.0124
    assert df["Video1_Perpetrator"][1] == 9.3277
    assert df["Video1_Perpetrator"][2] == 11.1553
    assert df["Video1_Perpetrator"][3] == 11.6845
    assert df["Video1_Perpetrator"][4] == 8.1307
    assert df["Video1_Perpetrator"][5] == 11.6240


def test_similarity_mtcnn_ghostfacenet_euclidean():
    WH_column_images = ImageLoader("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = ImageLoader("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImageAnalyzer(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="GhostFaceNet",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == 39.3741
    assert df["Video1_Perpetrator"][1] == 40.6091
    assert df["Video1_Perpetrator"][2] == 41.3193
    assert df["Video1_Perpetrator"][3] == 42.9959
    assert df["Video1_Perpetrator"][4] == 40.3540
    assert df["Video1_Perpetrator"][5] == 49.0519
