(ns opencvlab.core
  (:import [org.opencv.core Mat Size Point CvType MatOfKeyPoint Scalar Core TermCriteria]
           [org.opencv.highgui Highgui]
           [org.opencv.imgproc Imgproc]
           [org.opencv.features2d FeatureDetector DescriptorExtractor Features2d KeyPoint]))

(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)

(defn detect-keypoints [mat]
  (let [mser (FeatureDetector/create FeatureDetector/MSER)
        keypoints (MatOfKeyPoint.)]
    (.detect mser mat keypoints)
    keypoints))

(defn draw-keypoints! [mat keypoints result]
  (let [blue (Scalar. 255 0 0)
        random (Scalar/all -1)]
    (Features2d/drawKeypoints mat keypoints result random Features2d/DRAW_RICH_KEYPOINTS)))

(defn clone [mat]
  (.clone mat))

;; ----------------------------------------------------------------
;; Imgproc
;; ----------------------------------------------------------------

(defn blur 
  [img size]
  (let [blurred (clone img)]
    (Imgproc/blur img blurred (Size. size size) (Point. -1 -1) Imgproc/BORDER_DEFAULT)
    blurred))

(defn threshold 
  [img threshold max]
  (let [result (clone img)]
    (Imgproc/threshold img result threshold max Imgproc/THRESH_BINARY)
    result))

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

(defn is-keypoint? [x] (instance? KeyPoint x))

(defn kp-seq 
  [kp-mat]
  {:pre [(instance? MatOfKeyPoint kp-mat)]}
  (seq (.toArray kp-mat)))

(defn kp-mat [kps]
  {:pre [(seq? kps) (every? is-keypoint? kps)]}
  (let [m (MatOfKeyPoint.)
        a (into-array kps)]
    (.fromArray m a)
    m))

;; Naive O(n^2) implementation
(defn kp-dists 
  [keypoints]
  {:pre [(seq? keypoints) (every? is-keypoint? keypoints)]}
  (let [kps (map to-map keypoints)]
    (for [a kps
          b kps] 
      {:a a
       :b b
       :dist (dist (to-xy a) (to-xy b))})))

(defn nearby? [kpd]
  (let [max-radius-factor 7/4
        min-size (min (:size (:a kpd)) (:size (:b kpd)))
        max-dist (* max-radius-factor min-size)]
    (< (:dist kpd) max-dist)))

(defn similar-size? [kpd]
  (let [ratio (/ (:size (:a kpd)) (:size (:b kpd)))]
    (< 3/5 ratio 5/3)))

(defn non-trivial-size? 
  [kpd]
  {:pre [(:size kpd)]}
  (< 5 (:size kpd)))

(defn same-word-pairs 
  "Find neighbouring keypoints, a heuristic for letters belonging to the same word. Returns keypoint-dist maps."
  [keypoints]
  {:pre [(seq? keypoints) (every? is-keypoint? keypoints)]}
  (let [kpds (kp-dists keypoints)]
    (filter (fn [kpd] 
              (and (< 0 (:dist kpd))
                   (non-trivial-size? (:a kpd))
                   (non-trivial-size? (:b kpd))
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
        box (box-hull (enclosing-box (get p :a)) (enclosing-box (get p :b)))
        col (Scalar/all -1)]
    (Core/rectangle img (Point. (:x1 box) (:y1 box)) (Point. (:x2 box) (:y2 box)) col)))

(defn edges 
  "Get a map from each a to the set of each b connected to a."
  [pairs]
  {:pre [#(seq? pairs)]
   :post [(map? %)]}
  (->> 
   (group-by :a pairs)
   (map (fn [[k vs]] [k (map :b vs)]))
   (into {})))

(defn draw-hull! 
  [img hull]
  (let [col (Scalar/all -1)]
    (Core/rectangle img (Point. (:x1 hull) (:y1 hull)) (Point. (:x2 hull) (:y2 hull)) col)))
  

(defn overlapping-clusters 
  "Boxes around keypoints with near overlap."
  [img]
  (let [gray (clone img)
        keypoints (detect-keypoints gray)
        kps (kp-seq keypoints)
        result (clone gray)
        pairs (same-word-pairs kps)
        connected-to (edges pairs)
        kp-groups (map (fn [[from tos]] (set (conj tos from))) connected-to)
        enclosing-box-groups (map (fn [grp] (map enclosing-box grp)) kp-groups)
        hulls (map (fn [grp] (reduce box-hull grp)) enclosing-box-groups)
        distinct-hulls (set hulls)]
    (draw-keypoints! gray keypoints result)
    (doall 
     (for [x distinct-hulls] 
       (draw-hull! result x)))
    (imshow result)))


(defn adjacents [connected-to pt]
  (get connected-to pt))


(defn reachable-from 
  ([conn-to pt]
     (reachable-from conn-to #{pt} 16))
  ([conn-to pts max-depth]
     (if (<= max-depth 0)
       pts
       (let [adjs (apply clojure.set/union (map #(set (adjacents conn-to %)) pts))
             new (clojure.set/difference adjs pts)]
         (if (seq new)
           (reachable-from conn-to (clojure.set/union pts new) (dec max-depth))
           pts)))))


(defn reachable-clusters 
  "Boxes around keypoints reachable through near overlaps."
  [img]
  (let [gray (clone img)
        keypoints (detect-keypoints gray)
        kps (kp-seq keypoints)
        result (clone gray)
        ;; filter to a subset while we experiment
        subset (filter (fn [kp]
                         (and (<= (-> kp .pt .x) 600)
                              (<= 0 (-> kp .pt .y))
                              (<= 10 (.size kp) 40))) 
                       kps)
        pairs (same-word-pairs subset)
        connected-to (edges pairs)
        transitive-connections (->> connected-to
                                    (map (fn [[src adjacents]] [src (reachable-from connected-to src)])))
        kp-groups (map (fn [[from tos]] (set (conj tos from))) transitive-connections)
        unique-groups (set kp-groups)
        major-groups (take 50 (sort-by #(- (count %)) unique-groups))
        enclosing-box-groups (map (fn [grp] (map enclosing-box grp)) major-groups)
        hulls (map (fn [grp] (reduce box-hull grp)) enclosing-box-groups)
        distinct-hulls (set hulls)]
    (draw-keypoints! gray (kp-mat subset) result)
    (doall 
       #_(for [x transitive-connections] (println x))
       (for [x distinct-hulls] 
         (draw-hull! result x)))
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


  (let [img (clone mf)
        filtered (-> img clone (blur 5))
        keypoints (detect-keypoints filtered)
        result (clone filtered)]
    (draw-keypoints! filtered keypoints result)
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
