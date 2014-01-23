(ns opencvlab.core)

(import '[org.opencv.core Mat Size CvType MatOfKeyPoint Scalar]
        '[org.opencv.highgui Highgui]
        '[org.opencv.imgproc Imgproc]
        '[org.opencv.features2d FeatureDetector DescriptorExtractor Features2d])

(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)

(def input-file "resources/images/mf-2011_375x500.jpg")
(def output-file "resources/images/mf-2011_result.tif")

(def mf (Highgui/imread input-file 0))
(def mser (FeatureDetector/create FeatureDetector/MSER))
(def keypoints (MatOfKeyPoint.))
(.detect mser mf keypoints)

(def result (.clone mf))

(def blue (Scalar. 255 0 0))
(Features2d/drawKeypoints mf keypoints result blue 4)

(Highgui/imwrite output-file result)


;; ;; Converted from Java code at http://answers.opencv.org/question/23066/show-mat-image/
;; (defn to-buffered-image [mat]
;;   (let [out (Mat.)
;;         colour? (< 1 (.channels mat))
;;         type (if colour?
;;                java.awt.image.BufferedImage/TYPE_BYTE_GRAY
;;                java.awt.image.BufferedImage/TYPE_3BYTE_BGR)]
;;     (do
;;       (if colour? (Imgproc/cvtColor mat out Imgproc/COLOR_BGR2RGB))
;;       (let [blen (* (.channels mat) (.cols mat) (.rows mat))
;;             bytes (byte-array blen)]
;;         (.get out 0 0 bytes)
;;         (let [image (java.awt.image.BufferedImage. (.cols mat) (.rows mat) type)]
;;           (-> image
;;               .getRaster
;;               (.setDataElements 0 0 (.cols mat) (.rows mat) bytes))
;;           image)))))

;; (defn imshow [mat]
  
;;   )