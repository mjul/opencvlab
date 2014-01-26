(ns opencvlab.core
  (:import [org.opencv.core Mat Size Point CvType MatOfKeyPoint Scalar Core TermCriteria]
           [org.opencv.highgui Highgui]
           [org.opencv.imgproc Imgproc]
           [org.opencv.features2d FeatureDetector DescriptorExtractor Features2d KeyPoint]
           [org.opencv.flann ]))

(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)


(defn detect-keypoints [mat]
  (let [mser (FeatureDetector/create FeatureDetector/MSER)
        keypoints (MatOfKeyPoint.)]
    (.detect mser mat keypoints)
    keypoints))

(defn draw-keypoints! [mat keypoints result]
  (let [blue (Scalar. 255 0 0)
        random (Scalar/all -1)]
    (Features2d/drawKeypoints mat keypoints result random 4)))

(defn clone [mat]
  (.clone mat))

;; ----------------------------------------------------------------

(defmulti to-map class)

(defmethod to-map Point [p] 
  {:x (.x p) :y (.y p)})

(defmethod to-map KeyPoint [kp] 
  {:point (to-map (.pt kp))
   :size (.size kp)
   :angle (.angle kp)
   :response (.response kp)
   :octave (.octave kp)
   :class-id (.class_id kp)})

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


(defn dist [[x1 y1] [x2 y2]]
  (let [dx (- x2 x1)
        dy (- y2 y1)]
    (java.lang.Math/sqrt (+ (* dx dx) (* dy dy)))))

(defn to-xy [kp-map]
  ((juxt :x :y) (:point kp-map)))


;; Naive O(n^2) implementation
(defn kp-dists [keypoints]
  (let [kps (map to-map (.toArray keypoints))]
    (for [a kps
          b kps] 
      {:a a
       :b b
       :dist (dist (to-xy a) (to-xy b))})))

(defn nearby? [kpd]
  (let [max-radius-factor 5/3
        min-size (min (:size (:a kpd)) (:size (:b kpd)))
        max-dist (* max-radius-factor min-size)]
    (< (:dist kpd) max-dist)))

(defn similar-size? [kpd]
  (let [ratio (/ (:size (:a kpd)) (:size (:b kpd)))]
    (< 3/4 ratio 4/3)))

(defn same-word-pairs 
  "Find neighbouring keypoints, a heuristic for letters belonging to the same word. Returns keypoint-dist maps."
  [keypoints]
  (let [kpds (kp-dists keypoints)]
    (filter (fn [kpd] 
              (and (< 0 (:dist kpd))
                   ;; eliminate duplicates (a,b) (b,a)
                   (<= (get-in kpd [:a :point :x])
                      (get-in kpd [:b :point :x]))
                   (nearby? kpd)
                   (similar-size? kpd)))
            kpds)))

(defn enclosing-box
  "Given a keypoint-dist structure, return the box enclosing the keypoint."
  [kpd]
  (let [point (:point kpd)
        r (/ (max (:size kpd)) 2)
        x1 (- (:x point) r)
        y1 (- (:y point) r)
        x2 (+ (:x point) r)
        y2 (+ (:y point) r)]
    {:x1 x1 :y1 y1, :x2 x2 :y2 y2}))

(defn box-hull 
  "Get the smallest box that encloses the two boxes a and b."
  [a b]
  {:x1 (min (:x1 a) (:x1 b))
   :y1 (min (:y1 a) (:y1 b))
   :x2 (max (:x2 a) (:x2 b))
   :y2 (max (:y2 a) (:y2 b))})

(defn draw-box-for-pair! [img p]
  (let [a (get-in p [:a :point])
        b (get-in p [:b :point])
        box (bounding-box (enclosing-box (get p :a)) (enclosing-box (get p :b)))
        col (Scalar/all -1)]
    (Core/rectangle img (Point. (:x1 box) (:y1 box)) (Point. (:x2 box) (:y2 box)) col)))

(defn edges 
  "Get a map from each a to the set of each b connected to a."
  [pairs]
  (->> 
   (group-by :a pairs)
   (map (fn [[k vs]] [k (map :b vs)]))
   (set)))

(defn draw-hull! 
  [img hull]
  (let [col (Scalar/all -1)]
    (Core/rectangle img (Point. (:x1 hull) (:y1 hull)) (Point. (:x2 hull) (:y2 hull)) col)))
  
(defn bigram-clusters [img]
  (let [gray (clone img)
        keypoints (detect-keypoints gray)
        result (clone gray)
        dists (kp-dists keypoints)]
    (draw-keypoints! gray keypoints result)
    (doall 
     (let [pairs (same-word-pairs keypoints)
           connected-to (edges pairs)
           kp-groups (map (fn [[from tos]] (set (conj tos from))) connected-to)
           enclosing-box-groups (map (fn [grp] (map enclosing-box grp)) kp-groups)
           hulls (map (fn [grp] (reduce box-hull grp)) enclosing-box-groups)
           distinct-hulls (set hulls)]
       (for [x distinct-hulls] 
         (draw-hull! result x))))
    (imshow result)))


;;
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
        result (clone gray)
        dists (kp-dists keypoints)]
    (draw-keypoints! gray keypoints result)
    (doall 
     (for [p (take 99999 (same-word-pairs keypoints))]
       (draw-box-for-pair! result p)))
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
    ;; (Imgproc/blur mf blurred (Size. 15 15) (Point. -1 -1) Imgproc/BORDER_DEFAULT)

  (let [blurred (clone mf)]
    (Imgproc/blur mf blurred (Size. 15 15) (Point. -1 -1) Imgproc/BORDER_DEFAULT)
    (let [mask (clone blurred)]
      (Imgproc/threshold blurred mask (double 80) 255 Imgproc/THRESH_BINARY)
      (imshow mask)))

)
