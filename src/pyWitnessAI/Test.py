from .VideoAnalyzer import *


def test():
    #  Input the path and arguments needed
    video_path = "E:/Project.Pycharm/FaceDetection/Face_detection/Materials/2faces.mp4"
    #  cascade_path = 'E:/Project.Pycharm/FaceDetection/Face_detection/Models/haarcascade_frontalface_alt.xml'
    analyzer = VideoAnalyzer(video_path)
    # lineup_number = 2

    mtcnn_analyzer = FrameAnalyzerMTCNN()
    opencv_analyzer = FrameAnalyzerOpenCV()
    analyzer.add_analyzer(mtcnn_analyzer)
    analyzer.add_analyzer(opencv_analyzer)

    #  Load the lineup
    lineup_loader = LineupLoader(
                                 image_path=['E:/Project.Pycharm/FaceDetection/Assessment/Lineup/Damien99.jpg',
                                             'E:/Project.Pycharm/FaceDetection/Assessment/Lineup/Damien1999.jpg'])
    lineup = lineup_loader.load_image()

    #  Comparison of lineup faces with faces in the video
    similarity_analyzer = SimilarityAnalyzer(lineup_images=lineup, face_detector="mtcnn")
    analyzer.add_analyzer(similarity_analyzer)

    analyzer.run(0, 10)

    print(analyzer.get_analysis_info())

    analyzer.save_data_flattened()

    plt.figure(figsize=(9, 6))
    plt.suptitle("Video analysis", fontsize=24, fontweight='bold')
    plt.subplot(2, 2, 1)
    analyzer.plot_face_counts()
    plt.subplot(2, 2, 2)
    analyzer.plot_face_areas()
    plt.subplot(2, 2, 3)
    analyzer.plot_average_pixel_values()
    plt.subplot(2, 2, 4)
    analyzer.plot_mtcnn_confidence_histogram()

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()
