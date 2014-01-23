(ns opencvlab.core
  (:import [org.opencv.core Mat Size Point CvType MatOfKeyPoint Scalar Core TermCriteria]
           [org.opencv.highgui Highgui]
           [org.opencv.imgproc Imgproc]
           [org.opencv.features2d FeatureDetector DescriptorExtractor Features2d]))

(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)


(defn detect-keypoints [mat]
  (let [mser (FeatureDetector/create FeatureDetector/MSER)
        keypoints (MatOfKeyPoint.)]
    (.detect mser mat keypoints)
    keypoints))

(defn draw-keypoints! [mat keypoints result]
  (let [blue (Scalar. 255 0 0)]
    (Features2d/drawKeypoints mat keypoints result blue 4)))

(defn clone [mat]
  (.clone mat))

;; ----------------------------------------------------------------
;; Helper functions to show the results
;; ----------------------------------------------------------------

;; Converted from Java code at http://answers.opencv.org/question/23066/show-mat-image/
(defn to-buffered-image [mat]
  (let [out (Mat.)
        colour? (< 1 (.channels mat))
        type (if colour?
               java.awt.image.BufferedImage/TYPE_3BYTE_BGR
               java.awt.image.BufferedImage/TYPE_BYTE_GRAY)
        width (.cols mat)
        height (.rows mat)]
    (do
      (if colour?
        (Imgproc/cvtColor mat out Imgproc/COLOR_BGR2RGB)
        (.copyTo mat out))
      (let [blen (* (.channels mat) width height)
            bytes (byte-array blen)]
        (.get out 0 0 bytes)
        (let [image (java.awt.image.BufferedImage. width height type)
              raster (.getRaster image)]
          (.setDataElements raster 0 0 width height bytes)
          image)))))

(defn show-frame [image]
  (doto (javax.swing.JFrame.)
    (.setTitle "Hello, world")
    (.add (proxy [javax.swing.JPanel] []
            (paint [g] (.drawImage g image 0 0 nil))))
    (.setSize (java.awt.Dimension. (.getWidth image) (.getHeight image)))
    (.show)))

(defn imshow [mat]
  (show-frame (to-buffered-image mat)))

;; ----------------------------------------------------------------
;; REPL PLAYGROUND
;; ----------------------------------------------------------------

(comment

  (def input-file "resources/images/mf-2011_375x500.jpg")
  (def output-file "resources/images/mf-2011_result.tif")
  (def mf (Highgui/imread input-file org.opencv.highgui.Highgui/CV_LOAD_IMAGE_GRAYSCALE))

  ;;(Highgui/imwrite output-file result)

  (let [gray (clone mf)
        keypoints (detect-keypoints gray)
        result (clone gray)]
    (draw-keypoints! gray keypoints result)
    (imshow result))

  
  (let [kmresult (clone mf)
        labels (clone keypoints)
        k 10
        crit (TermCriteria. (+ TermCriteria/COUNT TermCriteria/EPS) 10 1.0)
        attempts 4
        flags Core/KMEANS_RANDOM_CENTERS]
      )

    ;; (Imgproc/threshold mf dst (double 120) 255 Imgproc/THRESH_BINARY)
    ;; (Imgproc/adaptiveThreshold mf dst (double 120) Imgproc/ADAPTIVE_THRESH_MEAN_C Imgproc/THRESH_BINARY 5 5)

  (let [blurred (clone mf)]
    (Imgproc/blur mf blurred (Size. 10 10) (Point. -1 -1) Imgproc/BORDER_DEFAULT)
    (imshow blurred)
    )