// OpenGL
#define GL_GLEXT_PROTOTYPES  // preprocessor macro definition ; the macro preprocessor is a powerful but primitive text‐substitution step that gives you file inclusion, conditional compilation, and simple “find-and-replace” macros.
#include <GL/gl.h> // core OpenGL API
#include <GL/glu.h> // OpenGL utility library
#include <GLFW/glfw3.h>
/*
glfw.h inclues header for GLFW, lightweight cross platform library for creating, 
 - OpenGL context and window
 - Processing user input
 - Managing the main loop and buffer swapping
Allows to call functions like glfwInit(), glfwCreateWindow(), and glfwPollEvents().
*/


// OpenCV
#include <opencv2/opencv.hpp>

// g2o general graph optimization  - Non linear framework For Bundle Adjustment
// tells g2o to compile against the copy of Ceres Solver that comes bundled (“vendored”) with g2o, rather than expecting me to link an external Ceres library. This can simplify dependencies if i don’t already have Ceres installed.
#define G2O_USE_VENDORED_CERES 
// Declares the BlockSolver template class, which manages the partitioning of our problem into “blocks” of variables (e.g. 6‐DoF poses and 3D points) and sets up the Schur‐complement trick for efficient linearization. // filters degenerates and helps to reduce search space
#include <g2o/core/block_solver.h> 
// Defines OptimizationAlgorithmLevenberg, a Levenberg–Marquardt wrapper that drives the iterative solve: it linearizes our error terms, forms normal equations, applies damping, and updates the estimate each round.
#include <g2o/core/optimization_algorithm_levenberg.h> 
// Provides outlier‐resistant “robust kernels” (e.g. Huber, Cauchy). We attach one to each edge so that large reprojection errors (from bad correspondences) get down‐weighted rather than derailing the entire optimization.
#include <g2o/core/robust_kernel_impl.h> 
// Supplies a LinearSolverEigen implementation that uses Eigen’s sparse linear algebra routines under the hood. The block solver needs a linear solver to invert or factor the Hessian‐approximation matrix at each LM step.
#include <g2o/solvers/eigen/linear_solver_eigen.h> 

/*
Pulls in the ready-made types for “six‐degree‐of‐freedom with exponential map”:
- VertexSE3Expmap for camera poses (rotation + translation).
- EdgeSE3ProjectXYZ for projecting 3D points into a pose, measuring reprojection error.
- VertexPointXYZ for the 3D landmarks themselves.
*/
#include <g2o/types/sba/types_six_dof_expmap.h>

// Standard
#include <chrono> // timing tools
#include <cstdio> // Printf style logging
#include <unordered_map>  // a hash‐table–based associative container that offers average O(1) lookup, insertion, and deletion by key.
#include <vector>

class Frame {
public:
    // Static: One shared counter across all Frame instances; Assign unique id to every new frame
    static inline int id_generator = 0; 
    // saves this frame's unique identifier set in constructor form 
    int id;  
    // grayscale image matrix
    cv::Mat image_grey;
    // camera intrinsics K
    cv::Matx33d K;
    // Distorting coefficients (radius or tangential) for undistoring 2D points
    cv::Mat dist;
    // List of detected feature locations (with size & orientation) in the image
    std::vector<cv::KeyPoint> kps;
    // Corresponding descriptors
    cv::Mat des;
    // Pose of this camera in the world; a 3x3 rotation and 3x1 translation vector
    cv::Matx33d rotation;
    cv::Matx31d translation;
public:
    // Compiler generate ctor(constructor); leaves members uninitialized (to fill them later)
    Frame() = default;
    // Main parameterized constructor 
    Frame(const cv::Mat& input_image_grey, const cv::Matx33d& input_K, const cv::Mat& input_dist) {
        // One ORB detector/descriptor instance shared across all frames
        static cv::Ptr<cv::ORB> extractor = cv::ORB::create();
        // Assign the current frame's id, then increment the global counter
        this->id = Frame::id_generator++;
        // copy input's into object attributes similar to self logic
        this->image_grey = input_image_grey;
        this->K = input_K;
        this->dist = input_dist;
        // decalres a vector -  corners
        std::vector<cv::Point2f> corners;
        // Shi-Tomasi corner detection with (upto 3000 strong corners, 0.01 quality level, 7 px minimum distance)
        cv::goodFeaturesToTrack(
                image = image_grey, 
                corners = corners,
                maxCorners = 3000,
                quality =  0.01,
                minDist =  7);
        // Reserves memory in kps so that it can hold atleast corners.size() elements without needig to re-allocate ensurings single allocation
        this->kps.reserve(corners.size());
        // iterates over each element of the corners array by reference (avoiding a copy)
        // Each corner hold an (x,y) coordinate in the image where the feature is detected
        for (const cv::Point2f& corner : corners) {
            // Creates a keypoint centered at (corner.x corner.y) with a patch size (diameter of keypoint) of 20 pixels.
            // Append new kp at end of kps vector cuz previously reserved enough space avoids reallocation
            this->kps.push_back(cv::KeyPoint(pt = corner, _size = 20));
        }
        // Compuet ORB descriptors for all keypoints and store them in des.
        extractor->compute(image_grey, this->kps, this->des);
        // Initialize camera pose to "identity" (i.e. at the world origin looking down Z)
        this->rotation = cv::Matx33d::eye();
        this->translation = cv::Matx31d::zeros();
        // Log how many features were extracted in this frame
        std::printf("Detected: %zu features\n", this->kps.size());
    }
};

// Instances of this class represent one 3D point in our map
class Landmark {
    // Everything that follows the next access specifier is public
    public:
        // Shared by all Landmark objects
        static inline int id_generator = 0;
        int id;
        cv::Matx31d location;
        cv::Matx31d colour;
    public:
        Landmark() = default;
        Landmark(const cv::Matx31d& input_location, const cv::Matx31d& input_colour) {
            // Class-wide (across all instances) postfix incremement 
            // Postfix : returns old value of x and then increments
            // starts with first id as 0, but if we need to start with 1, then ++Landmark::id_generator should help
            this->id = Landmark::id_generator++; // specifying the Landmark class non-static id that we have declared earlier
            this->location = input_location;
            this->colour = input_colour;
        }
};


class Map {
public:
    // an hash-map from each Frame's unique id to the Frame object
    std::unordered_map<int, Frame> frames;
    std::unordered_map<int, Landmark> landmarks;
    // **Key value mapping**, Maps each 3D point's id tp its landmark
    // observations: for each landmark ID, stores a vector pairs (this landmark was seen in that frame at that keypoint)
    std::unordered_map<int, std::vector<std::pair<int, int>>> observations; // key is landmark_id, pair are frame_id and kp_index
    //             landmark ID  └─ list of (frame ID, keypoint index)

public:
    // Takes in a frame by reference then copies it into the frames map under its own id key
    void add_frame(const Frame& frame) {
        frames[frame.id] = frame;
    }
    // Takes in a landmark by reference then copies it into the landmark map under its own id key
    void add_landmark(const Landmark& landmark) {
        this->landmarks[landmark.id] = landmark;
    }
    // insert 3D point 
    void add_observation(const Frame& frame, const Landmark& landmark, int kp_index) {
        this->observations[landmark.id].push_back({ frame.id, kp_index });
    }
    // Triangulation
    // Can call Map::triangulate(...) without an instance of Map
    // Returns a 4x1 homogeneous coordinate (X, Y, Z, W) stored in cv::Matx41d
    static cv::Matx41d triangulate(const Frame& lhs_frame, const Frame& rhs_frame, const cv::Point2d& lhs_point, const cv::Point2d& rhs_point) {
        // A 3x4 matrix [R | t] for the first camera: 
        // Matx34d stores exactly 3 rows x 4 columns of type double
        cv::Matx34d cam1 = {
            lhs_frame.rotation(0,0), lhs_frame.rotation(0,1), lhs_frame.rotation(0,2), lhs_frame.translation(0),
            lhs_frame.rotation(1,0), lhs_frame.rotation(1,1), lhs_frame.rotation(1,2), lhs_frame.translation(1),
            lhs_frame.rotation(2,0), lhs_frame.rotation(2,1), lhs_frame.rotation(2,2), lhs_frame.translation(2),
        };
        // Build second camera's projection the same way 3x4
        cv::Matx34d cam2 = {
            rhs_frame.rotation(0,0), rhs_frame.rotation(0,1), rhs_frame.rotation(0,2), rhs_frame.translation(0),
            rhs_frame.rotation(1,0), rhs_frame.rotation(1,1), rhs_frame.rotation(1,2), rhs_frame.translation(1),
            rhs_frame.rotation(2,0), rhs_frame.rotation(2,1), rhs_frame.rotation(2,2), rhs_frame.translation(2),
        };

        // Normalized points lhs (cam1)
        std::vector<cv::Point2d> lhs_point_normalised;
        // Undistort 
        // Takes distorted image coordinates and undistorted coordinates (ideal pinhole camera model)
        // Uses K and distortion coeffs to produce normalized, undistorted coords
        cv::undistortPoints(src = std::vector<cv::Point2d>{lhs_point}, lhs_point_normalised, lhs_frame.K, lhs_frame.dist);
        // Normalized points rhs (cam2)
        std::vector<cv::Point2d> rhs_point_normalised;
        cv::undistortPoints(std::vector<cv::Point2d>{rhs_point}, //crc
                            rhs_point_normalised,  // dst
                            rhs_frame.K, // cameraMatrix
                            rhs_frame.dist, // distCoeffs
                            cv::noArray(), // R
                            cv::noArray()); // T


        cv::Mat potential_landmark(rows = 4,cols= 1, type = CV_64F);

        // triangulatePoints: implements linear triangulation (e.g. SVD or DLT), solving:
        //   [ p1 × (P1 X)
        //     p2 × (P2 X) ] = 0,
        // where P1 = [R1 | t1] and P2 = [R2 | t2], and returns X in homogeneous form.
        cv::triangulatePoints(
                        cam1,  // rojMatr1 
                        cam2,  // projMatr2
                        lhs_point_normalised, //  projPoints1 
                        rhs_point_normalised, // projPoints2
                        potential_landmark); // points4D

        /*
        OpenCV’s triangulatePoints uses the Direct Linear Transform (DLT) 
        formulation and then solves the resulting homogeneous linear system via 
        Singular Value Decomposition (SVD). In other words, there’s a single fixed algorithm 
        under the hood: you build the DLT matrix A so that A·X = 0 and then compute the null‐space 
        of A using SVD to recover X.
        */

        // Can be converted to Euclidean (X/W, Y/W, Z/W)

        /* equivalent 
        np.array([
        potential_landmark[0, 0],
        potential_landmark[1, 0],
        potential_landmark[2, 0],
        potential_landmark[3, 0]
        ], dtype=np.float64)
        */

        return cv::Matx41d{
            // .at<double>(0) directly accesses the stored double element (no runtime type conversion)
            potential_landmark.at<double>(0),   // accessing element at a specified index pos
            potential_landmark.at<double>(2),
            potential_landmark.at<double>(3)
        };
    }
    // <opencv2/video/tracking.hpp>

    void optimise(int local_window, // controls how many recent frames to include in this BA
                 bool fix_landmarks, // bool: if true keeps all 3D points fixed(and only optimizes camera poses)
                 int rounds         // number of Levenberg-Marquardt iterations to run
                ) {
        // Creates the linear solver that g2o will use under the hood (Eigen's Sparse Cholesky)
        // unique_pt<...> manages the memory automatically
        // Wraps the linear solver into a blocksolver that handles "6DoF pose+3D point" blocks
        // std::move(...) transfer ownership into the block solver constructor.

        /*
         - Declares solver_ptr as a smart pointer that solely owns a BlockSolver instance, when solver goes out of scope, it will automatically delete its pointees.  By using unique_ptr everywhere, there’s no confusion about who must call delete
         - std::make_unique allocates and constructs the BlockSolver on the heap
         */

        // Block Solver needs linear-solver object to do its inner matrix factorizations
        // BlockSolver_6_3::PoseMatrixType is a typedef for an Eigen 6×6 matrix
        // It represents the Hessian/block size for camera‐pose variables (which have 6 DoF)
        // We pass it to LinearSolverEigen<> so g2o knows to build and solve linear systems of size 6
        // Each camera pose is treated as a 6-Dimensional variable
        // The Pose Transformation has 6 DOF ( 3 rotation and 3 translation)
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
        // Chooses the Levenberg–Marquardt strategy (damped Gauss–Newton) to optimize the whole graph.
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        solver->setWriteDebug(false); // Disables g2o’s verbose debug output.
        // Creates the optimizer object opt and tells it “use the solver we just built”.
        g2o::SparseOptimizer opt;
        opt.setAlgorithm(solver);
        // PoseMatrix as linear solver -> ownership to BlockSolver -> use solver

        // Add frames.
        // 1) Counters and window limits
        int non_fixed_poses = 0; //  will count how many poses you leave free to be optimized.
        int landmark_id_start = 0; // used later to assign unique IDs to 3D‐point vertices
        // computes the lowest frame index you still want to include when doing a local bundle adjustment of size local_window.
        const int local_window_below = Frame::id_generator - 1 - local_window; 
        // shifts that boundary by one (or two if you’re also fixing landmarks) to decide which older poses to lock (i.e. not optimize) so the window slides correctly.
        const int local_window_fixed_below = local_window_below + 1 + (fix_landmarks == false);
        //  iterate over all stored frames (frame_id is the key, frame the data).
        for (const auto& [frame_id, frame] : this->frames) {
            //  (i.e. doing local BA) and this frame’s id is older than the window bottom, continue and skip adding it altogether.
            // Filters frames to include only those in your local BA window.
            if ((local_window > 0) && (frame_id < local_window_below)) {
                continue;
            }
            // 2) Convert OpenCV pose --> g2o SE3Quat // Spatial Euclidean Group SE(3) --> set of all 3D rotations + translations
            // frame.rotation is a CV Matx (row-major), but Eigen wants column-major, so you take its transpose (.t()) and then map its raw .val array into an Eigen::Matrix3d.
            const g2o::Matrix3 rotation = Eigen::Map<const Eigen::Matrix3d>(frame.rotation.t().val); // Eigen is column major.
            // 
            const g2o::Vector3 translation = Eigen::Map<const Eigen::Vector3d>(frame.translation.val);
            // pack both into a g2o::SE3Quat, the standard g2o representation of a 6-DOF pose.
            g2o::SE3Quat se3(rotation, translation); // quaternion
            // Create and configure the vertex; Converts each selected frame’s (R, t) into a g2o pose vertex.
            g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap(); //allocates raw memory; Allocate a new g2o vertex (VertexSE3Expmap).
            // Set its initial guess (setEstimate) to the pose that we just mapped.
            v_se3->setEstimate(se3);
            // Give it a unique integer ID (we reuse frame_id here, so pose IDs don’t conflict with landmark IDs).
            v_se3->setId(frame_id);
            // Adds filtered and fixed poses to the optimizer and tracks a safe ID offset for landmarks.
            if ((frame_id <= (fix_landmarks == false)) || ((local_window > 0) && (frame_id < local_window_fixed_below))) {
                v_se3->setFixed(true);
            }
            else {
                ++non_fixed_poses; // increment fo non fixed poses
            }
            // 4) registers the pose vertex with the optimizer
            opt.addVertex(v_se3);
            if (frame_id > landmark_id_start) {
                landmark_id_start = frame_id + 1;
            }
        }

        // *****************Add landmarks*********************
        // keep track of how many landmark vertices and measurement edges we added that are not fixed.
        int non_fixed_landmarks = 0; 
        int non_fixed_edges = 0;
        // loop over every landmark in the map
        // .py equivalent: ```for (landmark_id, landmark) in self.landmarks.items()```
        // this->landmarks is an unordered_map<int, Landmark> *unpack into landmark_id(key) and landmark (the Landmark object)
        for (const auto& [landmark_id, landmark] : this->landmarks) {
            // 2a) Decides whether this landmark lies within the local window of recent frames
            // In local BA we only optimize points seen in a certain range of recent frames
            bool in_local = false;
            // We scan observations[landmark_id] which lists all (frame_id, kp_index) pairs where this landmark was seen
            // .py equivalent : ```for frame_id, kp_index in self.observations[landmark_id].items():```
            for (const auto& [frame_id, kp_index] : this->observations[landmark_id]) { 
                // If local_window ==0 , we treat everything as "in local"
                // Otherwise check that at least one of those frame_id values is ≥ local_window_below (i.e.within our sliding window)
                if ((local_window == 0) || ((frame_id >= local_window_below))) {
                    in_local = true;
                    break;
                }
            }
            // If none are recent, we continue and skip the landmark entirely
            if (!in_local) {
                continue;
            }
            /*
            this->observations[landmark_id] --> // Use operator[] when you intend to insert a default element if the key might be missing (e.g. building up your observations as you go).

            this->landmarks.at(landmark_id) --> // Use .at() when you want to access something you know must already exist, and you’d rather get an exception than accidentally create a bogus entry.
                                                // Will std::out_of_range error if doen't exist
            */
            // .py eq: cvpt = self.landmarks[landmark_id].location
            const cv::Matx31d& cvpt = this->landmarks.at(landmark_id).location;
            // For eg: cvpt = np.array([x, y, z], dtype=np.float64)  # your 3×1 point; pt = np.array([cvpt[0], cvpt[1], cvpt[2]], dtype=np.float64)
            // g2o expects its own Vector3 type, so we copy the three coordinates.
            const g2o::Vector3 pt(cvpt(0), cvpt(1), cvpt(2));   

            // Create a new graph vertex for this 3D point in space
            g2o::VertexPointXYZ* v_pt = new g2o::VertexPointXYZ();
            v_pt->setId(landmark_id_start + landmark_id); // each vertex in g2o must have a unique integer ID—here we offset by landmark_id_start so it doesn’t clash with pose‐vertex IDs.
            v_pt->setEstimate(pt); // initialize the optimizer's guess for this point to our current map coordinate
            v_pt->setMarginalized(true); //  tells g2o to marginalize (i.e. handle via Schur complement) these point variables, which is standard practice in BA.
            v_pt->setFixed(fix_landmarks); // If fix_landmarks==true, we never move these points, otherwise they're free to adjust
            // 2b) Count how many of these points are actually free'
            //  If we are not fixing landmarks, we increment a counter so we know we have something to optimize.
            non_fixed_landmarks += (fix_landmarks == false); // fix_landmarks are ones that needs to be optimized
            // 2c) Finally add the vertex into the optimizer
            opt.addVertex(v_pt); // This hands ownership of the pointer to the g2o optimizer. From now on, g2o manages it.

            //********************* Add edges *****************************************

            // TL; DR
            /* Each iteration here ties one 3D landmark to one 2D keypoint observation. During optimization, g2o “knows”:
            1. Which 3D point (Vertex 0)
            2. Which camera pose (Vertex 1)
            3. Where in the image you saw it (measurement uv)
            4. How certain you are (information & robust kernel)
            5. How to project a 3D point through that camera (fx,fy,cx,cy)
            …and it adjusts all variables to make reprojections match those measurements as closely as possible.
            */

            // bundle-adjustment setup, adding one g2o “edge” per observation of a landmark in a frame:
            // cvpt holds the 3D coords (X, Y, Z) of the landmark whereas cvkp is the pixel location of one of ORB keypoints in a particular frame (u, v)
            // So in each iteration we take a 3D landmark position (cvpt) and its observed 2D keypoint projection (cvkp) and create a g2o edge that says “project this 3D point through that camera and it should land at these pixel coordinates.”
            // this->observations[landmark_id] is a std::vector of (frame_id, kp_index) pairs; Structured binding ([frame_id, kp_index]) pulls out each pair
            for (const auto& [frame_id, kp_index] : this->observations[landmark_id]) {
                // If you’re doing local BA (i.e. only frames within a sliding window), skip observations from frames too old.
                if ((local_window > 0) && (frame_id < local_window_below)) {
                    continue;
                }
                const Frame& frame = frames.at(frame_id); // Look up the Frame object by its frame_id.
                // Get the 2D pixel location (.pt) of the keypoint in that frame.; We cast the stored int kp_index to unsigned int to index
                const cv::Point2f& cvkp = frame.kps[static_cast<unsigned int>(kp_index)].pt;
                // Grab a reference to the camera’s intrinsic matrix 
                const cv::Matx33d& cvk = frame.K;
                // Create a new “projection” edge: it will measure a 3D point in space (Vertex 0) projected into a camera pose (Vertex 1) and compare to the observed pixel.
                g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
                // Vertex 0: the 3D point’s variable ID is landmark_id_start + landmark_id (we offset so IDs don’t clash with frame IDs).
                edge->setVertex(0, opt.vertex(landmark_id_start + landmark_id));
                // Vertex 1: the camera pose’s variable ID is exactly frame_id
                edge->setVertex(1, opt.vertex(frame_id));
                // Bookkeeping: count how many of these edges actually connect optimizable variables.
                if ((landmark_id >= 0)) && (frame_id > (fix_landmarks == false)) {
                    ++non_fixed_edges;
                }
                // Tell g2o “my observed pixel is (u, v) = (cvkp.x, cvkp.y)”. That becomes the measurement it will try to fit.
                edge->setMeasurement(g2o::Vector2(cvkp.x, cvkp.y));
                // The “information matrix” is the inverse of the measurement covariance. Here we say “all directions are equally certain, unit weight.”
                edge->setInformation(g2o::Matrix2::Identity());
                // Wrap this edge in a Huber robust kernel to reduce the impact of outliers (points that reprojection-error far exceed expectations).
                // .py eq: ```robust_kernel = RobustKernelHuber()```
                g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber(); // Hey, create a new Huber robust kernel instance and give me a pointer to it so I can use it later. 
        
                // Configure a Huber robust kernel to down-weight outliers:
                // - 5.991 is the 95%-quantile of the χ² (Chi-Sqyuare) distribution with 2 DOF (for 2D reprojection error).
                // - sqrt(5.991) ≈ 2.447 pixels: residuals ≤2.447px use squared error (inliers),
                //  residuals >2.447px use linear error (outliers get down-weighted). 
                // tells the Huber kernel “up to ±2.447 pixels of residual, keep using the regular squared error; beyond that, switch to a linear penalty so that gross outliers don’t blow up your total cost.”
                robust_kernel->setDelta(std::sqrt(5.991)); // δ = 2.447 is absolute residual threshold; This choice ensures that genuine inliers (with reprojection errors under about 2.4 px) are optimized as usual, while measurements with larger errors get down-weighted, improving robustness to mismatches.
                edge->setRobustKernel(robust_kernel);
                // Tell the edge the intrinsics so it can project 3D→2D properly:
                edge->fx = cvk(0,0); // focal length in pixels fx, fy
                edge->fy = cvk(1,1);
                edge->cx = cvk(0,2); // prinicpal points offsets cx,cy
                edge->cy = cvk(1,2);
                // Finally, register this edge with the optimizer. When we call opt.initializeOptimization() + opt.optimize(), g2o will adjust both the camera poses and the 3D point positions to minimize the sum of squared reprojection errors across all these edges.
                opt.addEdge(edge);
            }
        }

        // Now the g2o optimizer graph is built (vertices + edges)

        // Check for some invalid optimizer states.
        // If we have no vertices (no camera poses or points) or no edges (no reprojection constraints), there’s nothing to optimize.
        if (opt.vertices().empty() || opt.edges().empty()) {
            std::printf("Optimised: No edges to optimise [%zu %zu]\n", opt.vertices().size(), opt.edges().size());
            return;
        }
        // non_fixed_poses == 0: Means all camera vertices in our window are marked fixed, so nothing to adjust.
        // (fix_landmarks == false) && (non_fixed_landmarks == 0): If we intend to optimize landmarks (fix_landmarks == false) but there are zero unfixed landmark vertices, again nothing to do.
        // non_fixed_edges == 0: No edges actually connect free variables—that also means no optimization can happen.
        if ((non_fixed_poses == 0) || ((fix_landmarks == false) && (non_fixed_landmarks == 0)) || (non_fixed_edges == 0)) {
            std::printf("Optimised: No non fixed poses [%d %d %d]\n", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
            return;
        }

        // If any of these is true, you log the three counters and return early.
        
        // Run the optimisation - Bundle Adjustment
        opt.initializeOptimization(); // Prepares the internal data structures (like ordering, Hessian, etc.).
        opt.computeActiveErrors(); // Calculates the current total reprojection error across all active edges.
        const double initialChi2 = opt.activeChi2(); // Save the “chi-squared” error before optimization.
        //opt.setVerbose(true);
        opt.optimize(rounds); // Run Levenberg–Marquardt for the specified number of iterations (rounds).
        // Apply optimised vertices to frames and landmarks.
        for (const auto& [vertex_id, vertex] : opt.vertices()) {
            if (vertex_id < landmark_id_start) {
                // Adjust camera pose
                // --- This is a camera pose vertex v1 ---
                Frame& frame = frames.at(vertex_id);
                const g2o::VertexSE3Expmap* se3v = static_cast<const g2o::VertexSE3Expmap*>(vertex);
                // Map g2o’s Eigen rotation & translation back into our cv::Matx
                // frame.rotation[:, :] = v_pt.estimate().rotation().matrix().T  # .T is transpose in NumPy ; NumPy arrays are row-major, but in most cases you don't need to worry unless you deal with very low-level memory layouts.12
                // 3x3 Matrix
                Eigen::Map<Eigen::Matrix3d>(frame.rotation.val) = se3v->estimate().rotation().matrix().transpose(); // Eigen is column major.
                // 3x1 vector
                Eigen::Map<Eigen::Vector3d>(frame.translation.val) = se3v->estimate().translation();
            }
            else {
                // Adjust geometry 3D point
                // -- This is a landmark (3D point) vertex v0 ---
                // Reserved IDs [0 … landmark_id_start-1] for camera poses, and [landmark_id_start … ) for points.
                Landmark& landmark = this->landmarks.at(vertex_id - landmark_id_start);
                const g2o::VertexPointXYZ* v_pt = static_cast<const g2o::VertexPointXYZ*>(vertex);
                Eigen::Map<Eigen::Vector3d>(landmark.location.val) = v_pt->estimate();
            }
        }
        std::printf("Optimised: %f to %f error [%zu %zu]\n", initialChi2, opt.activeChi2(), opt.vertices().size(), opt.edges().size());
    }

    // Culling is removing unnecessary or irrelevant elements from a dataset to improve efficiency.
    // In 3D vision or mapping, it often means discarding bad or redundant frames, points, or observations to keep the system fast and accurate.
    void cull() {
        // how many landmarks before culling
        // .size() returns how many entries (map points) we currently have; We store it to later report how many we removed.
        const size_t landmarks_before_cull = this->landmarks.size();
        // Iterate over landmarks by iterator, so we can erase safely while looping
        // .py eq: ```it = iter(self.landmarks.items()) ; for landmark_id, landmark in it ```
        for (std::unordered_map<int, Landmark>::iterator it = this->landmarks.begin(); 
            // Access the current landmark’s ID and data
            it != this->landmarks.end();) {
            // We omit ++it in the `for` header because we’ll advance (or reset) it manually after possible erasure.
            const std::pair<int, Landmark>& landmark = *it; // *it dereferences the iterator to a pair<key, value>.
            // landmark.first is the landmark’s integer ID; landmark.second is the Landmark object.
            // observations maps each landmark ID → vector of (frame_id, keypoint_index).
            // .at(id) retrieves that vector (throws an exception if missing).
            const std::vector<std::pair<int, int>>& landmark_observations = this->observations.at(landmark.first);
            // check filter conditions
            // not_seen_in_many_frames: if the landmark appears in 4 or fewer frames total, it’s weakly observed.
            bool not_seen_in_many_frames = landmark_observations.size() <= 4;
            // not_seen_recently: either never observed (empty()), or last seen more than 7 frames ago.
            // Frame::id_generator is the next new frame’s ID, so subtracting 7 tells “7 frames ago”.
            bool not_seen_recently = landmark_observations.empty() || ((landmark_observations.back().first + 7) < Frame::id_generator);
            // If it’s both weakly seen AND stale, remove it outright
            // If a landmark is both poorly tracked and hasn’t been updated recently, we delete it: Remove its entry in observations; 
            // Erase from landmarks, which returns the next valid iterator → assigned back to it
            if (not_seen_in_many_frames && not_seen_recently) {
                this->observations.erase(landmark.first);
                it = this->landmarks.erase(it);
                continue;  // skip straight to the next iterator position
            }
            // Compute the landmark’s average reprojection error
            float reprojection_error = 0.0f;
            // We’ll accumulate error across every observation:
            for (const auto& [frame_id, kp_index] : landmark_observations) {
                // Each frame_id, kp_index pair tells us which frame and which keypoint saw this landmark; Fetch that Frame from frames map.
                const Frame& frame = this->frames.at(frame_id);
                // The measured 2D pixel location
                // frame.kps[kp_index].pt is a Point2f → actual pixel coordinates where ORB detected the feature.
                // result of optimization
                const cv::Matx21d measured = {static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.x), static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.y)};
                // Transformation from original set  of 3D points
                // Project the landmark’s 3D location back into this camera ; world2cam
                // take the 3D landmark (location) in world coordinates, Rotate & translate it into the camera frame: frame.rotation * X + frame.translation.
                // Multiply by intrinsic matrix K → yields (u·w, v·w, w).
                const cv::Matx31d mapped = frame.K * ((frame.rotation * landmark.second.location) + frame.translation);
                // Divide x,y by w → final reprojected pixel (u, v). 
                const cv::Matx21d reprojected{mapped(0) / mapped(2), mapped(1) / mapped(2)};
                // Accumulate Euclidean pixel error; cv::norm(...) computes √((dx)² + (dy)²).
                reprojection_error += static_cast<float>(cv::norm(measured - reprojected));
            }
            // Sum these errors, then divide by the number of observations → the mean reprojection error in pixels.
            reprojection_error /= static_cast<float>(landmark_observations.size());
            // If the average error is too large, remove the landmark
            // 5.991 is the χ² threshold for a 95% confidence ellipse in 2D (with 2 DOF). If beyond that, it’s an outlier → we erase as before.
            // If it’s acceptable, we manually ++it to move to the next entry.
            if (reprojection_error > 5.991f) {
                this->observations.erase(landmark.first);
                it = this->landmarks.erase(it);
                continue;
            }
            // Otherwise, keep it and advance the iterator
            ++it;
        }
        // Report how many points we removed
        const size_t landmarks_after_cull = this->landmarks.size();
        // Subtract the “after” count from the “before” count to know how many landmarks got purged.
        std::printf("Culled: %zu points\n", landmarks_before_cull - landmarks_after_cull);
    }
};

class SLAM {
public:
    Map mapp; //  declares a member variable mapp of type Map, which holds all frames, landmarks, and observations.

public:
    void process_frame(const cv::Matx33d& K, const cv::Mat& image_grey) {
        // Internally, Frame will detect ORB keypoints, compute descriptors, set frame.id from a static counter, and initialize its pose to identity.
        Frame frame(image_grey, K, cv::Mat::zeros(1, 4, CV_64F)); // a 1×4 zero vector for distortion coefficients.
        this->mapp.add_frame(frame); // mapp.add_frame(frame) inserts this new Frame into the map’s frames container keyed by frame.id
        
        // Nothing to do for the first frame.
        if (frame.id == 0) { // frame.id == 0 is true only for the very first image that we ever process.
            return; // return; exits process_frame early—there’s no “previous frame” to match against yet.            
        }
        // Get the most recent pair of frames.
        // Frame::id_generator is a static integer that was incremented when you created frame.
        // So the latest frame’s ID is id_generator-1, and the one before that is id_generator-2.
        Frame& frame_current = mapp.frames.at(Frame::id_generator - 1);
        // mapp.frames.at(key) looks up the Frame in the unordered_map by its ID—and returns it by reference so you can modify its pose later.
        Frame& frame_previous = mapp.frames.at(Frame::id_generator - 2);
        // Helper Lambda functions for filtering matches.
        // Before computing matches, we define two small functions (lambdas) to filter out bad matches: 
        // static → one copy shared by all SLAM instances ; constexpr → can be evaluated at compile time (though here mainly symbolic);  
        //  takes a reference to a vector of “knn” match lists (matches), each containing up to two DMatch objects.

        // .py eq of std::vector<std::vector<cv::DMatch>>& matches is ```matches: list[list[cv2.DMatch]]```
        constexpr static const auto ratio_test = [](std::vector<std::vector<cv::DMatch>>& matches) { 

            // Calls std::remove_if to find any entry where
                                      // 1. there is only one (or zero) neighbor, or
                                      // 2. the ratio of best‐to‐second‐best match distance exceeds 0.75 (Lowe’s ratio test).
            matches.erase(std::remove_if(matches.begin(), matches.end(), [](const std::vector<cv::DMatch>& options){
                return ((options.size() <= 1) || ((options[0].distance / options[1].distance) > 0.75f));
            }), matches.end()); //So the }), matches.end()); is simply closing out the call to erase( firstIt, lastIt ),
        };
        
        // implementing the “symmetry test” to keep only matches that agree in both directions:
        // matches1: matches from current→previous ; matches2: matches from previous→current, 
        // symMatches: output container for the mutually consistent matches.


        /*```python
        def symmetry_test(matches1: list[list[cv2.DMatch]], matches2: list[list[cv2.DMatch]], sym_matches: list[list[cv2.DMatch]]):
            sym_matches.clear()  # wipe out old results

            for match1 in matches1:
                for match2 in matches2:
                    if (match1[0].queryIdx == match2[0].trainIdx) and (match2[0].queryIdx == match1[0].trainIdx):
                        sym_matches.append([
                            cv2.DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance)
                        ])
                        break```
        */
        constexpr static const auto symmetry_test = [](const std::vector<std::vector<cv::DMatch>> &matches1, const std::vector<std::vector<cv::DMatch>> &matches2, std::vector<std::vector<cv::DMatch>>& symMatches) {
            symMatches.clear(); // Inside, the first thing we do is symMatches.clear(), wiping out any old results     
            // Iterate over each entry in matches1.   
            // Each *matchIterator1 is a std::vector<cv::DMatch> (the result of a knnMatch, usually size 2).     
            for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin(); matchIterator1!= matches1.end(); ++matchIterator1) {
                
                for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches2.begin(); matchIterator2!= matches2.end();++matchIterator2) {
                    // For each candidate in matches1, scan all entries in matches2 and the other way
                    if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                        // (*matchIterator1)[0].queryIdx is the index of the keypoint in the current frame.
                        // (*matchIterator1)[0].trainIdx is the index of its best match in the previous frame.
                        // Conversely, in matches2 we expect to see the same pair reversed: its best match’s queryIdx (in the “previous” frame) should equal the original trainIdx, and
                        // its trainIdx (in the “current” frame) should equal the original queryIdx. If both hold, the match is mutually consistent.
                        // We push a new one‐element vector of DMatch into symMatches, copying over the queryIdx, trainIdx, and distance.
                        symMatches.push_back({cv::DMatch((*matchIterator1)[0].queryIdx,(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance)});
                        break;
                    }
                }
            }
        };


        // Establish Correspondence --> Feature Matching

        // Compute matches and filter.
        // Create matcher; cv::Ptr<> is OpenCV’s reference-counted smart pointer; BFMatcher does brute-force matching of binary descriptors (ORB uses binary patterns).
        // cv::NORM_HAMMING tells it to compare descriptors by Hamming distance (count of differing bits).
        static cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        // Prepare containers for k-NN matches
        // Each “outer” vector will hold, for each descriptor in one frame, a small list of its two nearest neighbors in the other frame (the “cp” vs “pc” means current→previous and previous→current).
        std::vector<std::vector<cv::DMatch>> matches_cp;
        std::vector<std::vector<cv::DMatch>> matches_pc;
        // Run k-NN matching in both directions
        matcher->knnMatch(frame_current.des, frame_previous.des, matches_cp, 2); // best two matches k = 2
        matcher->knnMatch(frame_previous.des, frame_current.des, matches_pc, 2); // // Given a vector of two nn matches per descriptor:
        // Lowe's nearest neighbor filtering
        // ratio_test lambda we defined above, removes any match list whose best match is not sufficiently better than the second-best (typically d1/d2 > 0.75). This prunes ambiguous feature correspondences.
        // Lowe showed that this dramatically cuts down on false correspondences without losing many true ones.
        ratio_test(matches_cp);
        ratio_test(matches_pc);
        std::vector<std::vector<cv::DMatch>> matches;
        //  symmetry_test lambda compares the two lists and keeps only those matches where
        //    - A’s best match in B is B’s best match in A, and
        //    - The index mapping is consistent.

        symmetry_test(matches_cp, matches_pc, matches); // The result is a smaller set of high-confidence matches in matches.


        // Create final arrays of good matches.
        // Allocate final arrays for indices & 2D points
        // match_index_*: which keypoint index in each frame
        std::vector<int> match_index_current;
        std::vector<int> match_index_previous;
        // match_point_*: the actual pixel coordinates (as 2×1 double matrices)
        std::vector<cv::Matx21d> match_point_current;
        std::vector<cv::Matx21d> match_point_previous;
        for (const std::vector<cv::DMatch>& match : matches) {
            const cv::DMatch& m = match[0]; // Extract the best match from each pair
            // Store the keypoint indices and pixel locations
            match_index_current.push_back(m.queryIdx);
            match_point_current.push_back({static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.x), static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.y)});
            match_index_previous.push_back(m.trainIdx);
            match_point_previous.push_back({static_cast<double>(frame_previous.kps[static_cast<unsigned int>(m.trainIdx)].pt.x), static_cast<double>(frame_previous.kps[static_cast<unsigned int>(m.trainIdx)].pt.y)});
        }
        std::printf("Matched: %zu features to previous frame\n", matches.size());



        // Pose estimation of new frame.
        // We only run this “essential‐matrix + recoverPose” step for the very first motion (i.e. when we’ve got frame 0→frame 1). 
        // After that, we just copy the previous pose as an initialization.
        if (frame_current.id < 2) {
            cv::Mat E; // E will hold the essential matrix (3×3),
            cv::Mat R; // R will be the recovered rotation (3×3),
            cv::Mat t; // t will be the recovered translation vector (3×1).            
            int inliers = cv::recoverPose(match_point_current, // points1
                                         match_point_previous, // points2
                                         frame_current.K,      // cameraMatrix1
                                         frame_current.dist,   // distCoeffs1 ; distortion
                                         frame_previous.K,     // cameraMatrix2
                                         frame_previous.dist,  // distCoeffs2
                                         E, R, t,              // outputs
                                         cv::RANSAC,           // method
                                         0.999,                // confidence that the estimated pose is correct (99.99% indicates very high)
                                         1.0);                 // threshold: maximum allowed reprojection error to consider an inlier (in pixels)
            cv::Matx33d rotation = {        
                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
            };
            cv::Matx31d translation = {
                t.at<double>(0),
                t.at<double>(1),
                t.at<double>(2)
            };
            // Recover pose seems to return pose 2 to pose 1 rather than pose 1 to pose 2, so invert it.
            rotation = rotation.t(); // measured 
            translation = -rotation * translation;
            // Set the initial pose of the new frame.
            frame_current.rotation = rotation * frame_previous.rotation;
            frame_current.translation = frame_previous.translation + (frame_previous.rotation * translation);
            std::printf("Inliers: %d inliers in pose estimation\n", inliers);
        }
        // we just copy the previous pose as an initialization.
        else {
            // Set the initial pose of the new frame from the previous frame.
            frame_current.rotation = frame_previous.rotation;
            frame_current.translation = frame_previous.translation;
        }


        // Cache observations from the previous frame that are in this one.

        std::unordered_map<int, int> frame_previous_points; // key is kp_index and data is landmark_id.
        for (const auto& [landmark_id, landmark_observations] : this->mapp.observations) { // this->mapp.observations is the map from each landmark ID → a list of (frame_id, kp_index) pairs.
            // Loop over every observation of this landmark
            // Each entry says “this landmark appeared in frame frame_id at keypoint index kp_index.”
            for (const auto& [frame_id, kp_index] : landmark_observations) { // landmark_observations is the std::vector<std::pair<int,int>> of where that landmark has been seen.                
                // We only care about observations that happened in frame_previous (the frame right before the current one).
                // 
                if (frame_id == frame_previous.id) {
                    // Whenever a landmark was seen in that frame, we record:
                    frame_previous_points[kp_index] = landmark_id;
                }
            }
        // So now `frame_previous_points` lets us, in O(1), ask “if I have a keypoint index j in the previous frame, was that already part of my map? And if so, which landmark?”
        }

        /*python

        observations_of_landmarks = 0  # Fast way to ask: which keypoint matches a 3D landmark?

        # Loop over all matched keypoints
        for i in range(len(match_index_previous)):
            found_point = frame_previous_points.get(match_index_previous[i])
            if found_point is not None:
                # Add observation to the corresponding landmark
                self.mapp.add_observation(frame_current, self.mapp.landmarks[found_point], match_index_current[i])
                observations_of_landmarks += 1
        */

        // Find matches that are already in the map as landmarks.
        int observations_of_landmarks = 0; // We want a fast way to ask, “Given a keypoint index in the previous frame, which 3D landmark does it correspond to (if any)?”
        // Loop over all landmarks in the map
        // match_index_previous is the list of keypoint indices in the previous frame that matched up (via descriptor matching) with keypoints in the current frame.
        // We’ll iterate over each matched pair by index i. 
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            // does this matched keypoint in the previous frame already correspond to a known landmark?
            // find(...) looks up match_index_previous[i] in our cache.
            // If it isn’t in the map, this keypoint wasn’t in any landmark yet.
            auto found_point = frame_previous_points.find(match_index_previous[i]);     
            // If it isn’t in the map, this keypoint wasn’t in any landmark yet.      
            if (found_point != frame_previous_points.end()) {
                // Add observations to features that match existing landmarks.
                // We already have a Landmark object landmarks[found_point->second].
                // 
                this->mapp.add_observation(frame_current, this->mapp.landmarks[found_point->second], match_index_current[i]);
                ++observations_of_landmarks; //Increment observations_of_landmarks to keep track of how many existing landmarks got one more “vote” from this frame.
            }
        }
        /*
        - Data association: before we try to triangulate new points, we must first “track” the points we already know about.

        - By caching and looking up this reverse‐map, we efficiently add observations of existing landmarks into the factor graph.

        - Those new edges will be fed into the bundle adjustment (optimise()) so that the pose of the new frame and the 3D point positions can both be refined, taking into account these re‐observations.
        */
        std::printf("Matched: %d features to previous frame landmarks\n", observations_of_landmarks);



        // Optimise the pose of the new frame using the points that match the current map.
        // Run local bundle adjustment
        // Refine new frame's pose against the current map points:
        // 1 = include only the most recent frame and its observations
        // true = Keep all the landmarks fixed
        // Do 50 Levenberg-Marquardt iterations , This refines only the new frames pose to better fit the already-mapped points
        this->mapp.optimise(1, true, 50);
        // Remove landmarks that are eithe seen two few times or have high reprojection error. Keeps the map clean of bad points
        this->mapp.cull();
        // Search for matches to other frames by projection.
        int observations_of_map = 0; // Try to re‐associate existing map points by projecting them into the new frame:
        for (const auto& [landmark_id, landmark] : this->mapp.landmarks) {
            // Check landmark is infront of frame.
            //  Project the 3D point into pixel coordinates:
            const cv::Matx31d mapped = frame_current.K * ((frame_current.rotation * landmark.location) + frame_current.translation);
            // Divide by z to get (u,v) image coordinates.
            const cv::Matx21d reprojected{mapped(0) / mapped(2), mapped(1) / mapped(2)};
            //  Discard if it lands too close to the image border:
            //  ignore points that project off‐screen or within 10 px of the edge (so we can safely sample descriptors).
            if ((reprojected(0) < 10) || (reprojected(0) > image_grey.cols - 11) || (reprojected(1) < 10) || (reprojected(1) > image_grey.rows - 11)) {
                continue; // early exits if its in borders for the current landmark_id
            }


            /*python
            landmark_observations = self.mapp.observations[landmark_id]

            # Don’t re-add an observation if we’ve already matched this landmark to the new frame.
            if any(obs[0] == frame_current.id for obs in landmark_observations):
                continue
            */
            // Check it has not already been matched.
            const std::vector<std::pair<int, int>>& landmark_observations = this->mapp.observations.at(landmark_id);
            //  don’t re‐add an observation if we’ve already matched this landmark to the new frame.
            if (std::find_if(landmark_observations.begin(), landmark_observations.end(), [&frame_current](const std::pair<int, int>& landmark_observation){
                return (landmark_observation.first == frame_current.id);
            }) != landmark_observations.end()) {
                continue;
            }

            // test each *detected* feature in the current frame:
            // Find all detected features that are near the landmark reprojection.
            for (size_t i = 0; i < match_point_current.size(); ++i) {
                // Is it near enough?
                // Proximity test: must lie within 2 px of the reprojected location
                // cv::norm computes Euclidean distance; only consider features that lie very close (≤2 px) to where the map point projects.
                if (cv::norm(match_point_current[i] - reprojected) > 2) {
                    continue;
                }
                // Has it already been matched? 
                auto found_point = frame_previous_points.find(match_index_previous[i]);
                // Skip if that feature was already used to track a landmark:
                if (found_point != frame_previous_points.end()) { // ensure we don’t double‐use the same previous‐frame keypoint.
                    continue;
                }
                // ******Check similarity**********
                // Descriptor test: compare landmark’s descriptor v/s this feature’s

                // Fetch the stored landmark descriptor by indexing into the old frame’s des matrix.
                const cv::Mat& des_landmark = this->mapp.frames
                                                              .at(landmark_observations[0].first) // frame of first obs
                                                              .des.row(landmark_observations[0].second);
                // Fetch the current descriptor by row index.
                const cv::Mat& des_current = frame_current.des.row(match_index_current[i]);
                //  if the two descriptors are sufficiently similar (threshold = 64), we accept this as the same 3D point.
                if (cv::norm(des_landmark, des_current, cv::NORM_HAMMING) < 64) {
                    // If All tests passed → record the new observation
                    this->mapp.add_observation(frame_current, landmark, match_index_current[i]);
                    // Marks the old keypoint as used (= -1).
                    frame_previous_points[match_index_previous[i]] = -1;
                    ++observations_of_map;
                }
                // This “re‐projection” step boosts tracking robustness: even if a feature matcher missed some map points originally,
                // projecting known 3D points and then doing a small descriptor check recovers extra correspondences without requiring a
                // full descriptor‐to‐descriptor search.
            }
        /*
        - Local BA refines the new camera’s pose against the fixed map.

        - Cull removes outlier landmarks.

        - Re‐projection search projects each remaining 3D point into the image, then looks for a nearby feature whose descriptor matches.

        - Add new observations for those that pass proximity + descriptor tests.

        - Log how many map points is re‐associated.

        */
        
        }
        std::printf("Matched: %d features to map landmarks\n", observations_of_map);


        // Optimise the pose of the new frame using reprojected points from the current map.
        this->mapp.optimise(1, true, 50); // keep all 3D points fixed, only adjust the new frame's pose
        this->mapp.cull(); // Refines the new camera pose against the existing map points before you try to add brand-new ones.

        // Add all the remaing matched features.
        // 👉 ***** Triangulating Brand-New Points *********
        int new_landmarks = 0;
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            auto found_point = frame_previous_points.find(match_index_previous[i]);
            if (found_point == frame_previous_points.end()) { // Only proceed if this match didn’t correspond to an existing map point. i.e. *found_point = None*
                // New landmark to be triangulated. 

                // uses the two camera projection matrices plus the undistorted 2D observations to solve for a 4×1 homogeneous point [X, y, Z, W]
                cv::Matx41d point = Map::triangulate(frame_current, frame_previous, frame_current.kps[static_cast<unsigned int>(match_index_current[i])].pt, frame_previous.kps[static_cast<unsigned int>(match_index_previous[i])].pt);
                // Valid triangulation? Discard degenerate cases
                if (std::abs(point(3)) < 1e-5) { // point(3) is the homogeneous scale W. If it’s effectively zero, the intersection was ill-posed.
                    continue;
                }
                cv::Matx31d pt = {point(0) / point(3), point(1) / point(3), point(2) / point(3)}; // Normaluze / W
                // Reproject infront of both cameras?  Ensure the point lies in front of each camera
                const cv::Matx31d mapped_previous = frame_previous.K * ((frame_previous.rotation * pt) + frame_previous.translation);
                // Checks the Z coordinate is positive (in front of the camera).
                if (mapped_previous(2) < 0) {
                    continue;
                }
                const cv::Matx31d mapped_current = frame_current.K * ((frame_current.rotation * pt) + frame_current.translation);
                if (mapped_current(2) < 0) {
                    continue;
                }
                // Reprojection error is low? Check reprojection error is small in both views (previous and current)
                // Projects back into pixel space (u=x/z,v=y/z).
                const cv::Matx21d reprojected_previous = {mapped_previous(0) / mapped_previous(2), mapped_previous(1) / mapped_previous(2)};
                // Compares against the original matched keypoint positions; rejects if the Euclidean pixel‐error > 2 px.
                if (cv::norm(match_point_previous[i] - reprojected_previous) > 2) {
                    continue;
                }
                const cv::Matx21d reprojected_current = {mapped_current(0) / mapped_current(2), mapped_current(1) / mapped_current(2)};
                if (cv::norm(match_point_current[i] - reprojected_current) > 2) {
                    continue;
                }
                // Add it.
                // Build a grayscale “colour” value: Samples the gray‐value at the keypoint and normalizes to [0,1] for a “monochrome” color.
                double colour = static_cast<double>(image_grey.at<unsigned char>(frame_current.kps[static_cast<unsigned int>(match_index_current[i])].pt)) / 255.0;
                // Create & store the Landmark: Instantiate a new Landmark with position pt and that colour triple.
                Landmark landmark(pt, cv::Matx31d{colour, colour, colour});
                this->mapp.add_landmark(landmark); // inserts it into your map’s landmark container.
                // Record its observations in both frames
                this->mapp.add_observation(frame_previous, landmark, match_index_previous[i]); // links that 3D point to the two keypoints you just used, so future BA knows about it.
                this->mapp.add_observation(frame_current, landmark, match_index_current[i]);
                ++new_landmarks;
            }
        }
        std::printf("Created: %d new landmarks\n", new_landmarks);

        // *** Post Triangulation and clean up ****
        // Refine the pose of the new frame again.
        this->mapp.optimise(1, true, 50);
        this->mapp.cull();
        // Optimise the whole map; Run a larger local (BA) (include nore franes and points)
        this->mapp.optimise(40, false, 50);
        this->mapp.cull();

        // And output the new frame’s full 4×4 pose matrix
        // Print the map status and pose.
        std::printf("Map status: %zu frames, %zu landmarks\n", this->mapp.frames.size(), this->mapp.landmarks.size());
        std::printf(
            "[[% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]]\n",
            frame_current.rotation(0,0), frame_current.rotation(0,1), frame_current.rotation(0,2),  frame_current.translation(0),
            frame_current.rotation(1,0), frame_current.rotation(1,1), frame_current.rotation(1,2),  frame_current.translation(1),
            frame_current.rotation(2,0), frame_current.rotation(2,1), frame_current.rotation(2,2),  frame_current.translation(2),
            0.0, 0.0, 0.0, 1.0
        );
    }
};

class Display {
public:
    GLFWwindow* window = nullptr;
    GLuint image_texture;
    float image_frame[3] = { -0.9f, 0.9f, 0.4f };

public:
    ~Display() {
        glfwDestroyWindow(this->window);
        glfwTerminate();
    }

    Display(int width,int height) {
        if (!glfwInit()) {
            std::fprintf(stderr, "ERROR: GLFW failed to initialize");
            exit(1);
        }
        glfwWindowHint(GLFW_MAXIMIZED , GL_TRUE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        this->window = glfwCreateWindow(width, height, "slam", nullptr, nullptr);
        glfwSetFramebufferSizeCallback(this->window, [](GLFWwindow*, int new_width, int new_height){ glViewport(0, 0, new_width, new_height); });
        glfwSetWindowCloseCallback(this->window, [](GLFWwindow* window_closing) { glfwSetWindowShouldClose(window_closing, GL_TRUE); });
        glfwMakeContextCurrent(this->window);
        glGenTextures(1, &this->image_texture);
    }

    cv::Mat capture() {
        glfwMakeContextCurrent(this->window);
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        width = (width / 8) * 8;
        glReadBuffer(GL_FRONT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        cv::Mat pixels(height, width, CV_8UC3);
        glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels.data);
        return pixels;
    }

    void render(const Map& mapp) {
        glfwMakeContextCurrent(this->window);
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        glViewport(0, 0, width, height);
        // Get the last frame processed.
        const Frame& frame_last_processed = mapp.frames.at(Frame::id_generator - 1);
        // Calculate the width and height of the inset image.
        const float ratio_image = (static_cast<float>(frame_last_processed.image_grey.cols) / static_cast<float>(frame_last_processed.image_grey.rows));
        const float ratio_screen = (static_cast<float>(width) / static_cast<float>(height));
        float image_width = 0;
        float image_height = 0;
        if (width > height) {
            image_width = this->image_frame[2];
            image_height = ratio_screen * this->image_frame[2] / ratio_image;
        } 
        else {
            image_width = ratio_image * this->image_frame[2] / ratio_screen;
            image_height = this->image_frame[2];
        }
        // OpenGL configuration.
        glDepthRange(0, 1);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClearDepth(1.0);
        glShadeModel(GL_SMOOTH);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDepthFunc(GL_LEQUAL);
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Configuration for 2D rendering.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1, 1, 1);
        glDisable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        // Rendering the image to the image texture.
        cv::Mat image;
        cv::drawKeypoints(frame_last_processed.image_grey, frame_last_processed.kps, image);
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Rendering the image texture to the screen.
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(this->image_frame[0], this->image_frame[1]);
        glTexCoord2f(0, 1);
        glVertex2f(this->image_frame[0], this->image_frame[1] - image_height);
        glTexCoord2f(1, 1);
        glVertex2f(this->image_frame[0] + image_width, this->image_frame[1] - image_height);
        glTexCoord2f(1, 0);
        glVertex2f(this->image_frame[0] + image_width, this->image_frame[1]);
        glEnd();
        // Configuration for 3D rendering.
        glColor3f(0, 0, 0);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (static_cast<double>(width) / static_cast<double>(height)), 0.001, 1000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        // Helper functions.
        constexpr static const auto rt_to_transform = [](const cv::Matx33f& R, const cv::Matx31f& t){
            cv::Matx44f transform = {
                R(0,0), R(0,1), R(0,2), t(0),
                R(1,0), R(1,1), R(1,2), t(1),
                R(2,0), R(2,1), R(2,2), t(2),
                0.0f,  0.0f,  0.0f, 1.0f
            };
            return transform;
        };
        constexpr static const auto opengl_coordinate_system = [](const cv::Matx44f& mat){
            cv::Matx44f swap = {
                1.0f,  0.0f,  0.0f, 0.0f,
                0.0f, -1.0f,  0.0f, 0.0f,
                0.0f,  0.0f, -1.0f, 0.0f,
                0.0f,  0.0f,  0.0f, 1.0f
            };
            return swap * mat * swap.inv();
        };
        // Render from beind the current pose.
        glMultMatrixf(
            opengl_coordinate_system(
                rt_to_transform(
                    mapp.frames.at(Frame::id_generator-1).rotation, 
                    mapp.frames.at(Frame::id_generator-1).translation
                ).t()
            ).val
        );
        glTranslatef(0, 0, -5);
        // Data for rendering camera frustra.
        const int edges[12][2] = {{0, 1}, {0, 3}, {0, 4}, {2, 1}, {2, 3}, {2, 7}, {6, 3}, {6, 4}, {6, 7}, {5, 1}, {5, 4}, {5, 7}};
        const float verticies[8][3] = {{1.0f, -1.0f, -0.9f}, {1.0f, 1.0f, -0.9f}, {-1.0f, 1.0f, -0.9f}, {-1.0f, -1.0f, -0.9f}, {0.1f, -0.1f, -0.1f}, {0.1f, 0.1f, -0.1f}, {-0.1f, -0.1f, -0.1f}, {-0.1f, 0.1f, -0.1f}};
        // Render current camera.
        const float scale_pose = 0.2f;
        glLineWidth(4);
        glBegin(GL_LINES);
        glColor3f(0.5f, 0.5f, 0.0f);
        for (int e = 0; e < 12; ++e) {
            for (int v = 0; v < 2; ++v) {
                cv::Matx41f vertex = {
                    verticies[edges[e][v]][0] * scale_pose,
                    verticies[edges[e][v]][1] * scale_pose,
                    verticies[edges[e][v]][2] * scale_pose,
                    1
                };
                cv::Matx44f pose = opengl_coordinate_system(
                    rt_to_transform(
                        mapp.frames.at(Frame::id_generator-1).rotation, 
                        mapp.frames.at(Frame::id_generator-1).translation
                    ).inv()
                );
                vertex = pose * vertex;
                glVertex3f(vertex(0), vertex(1), vertex(2));
            }
        }
        glEnd();
        // Render other cameras.
        float scale_camera = 0.2f;
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);
        for (const auto& [frame_id, frame] : mapp.frames) {
            for (int e = 0; e < 12; ++e) {
                for (int v = 0; v < 2; ++v) {
                    cv::Matx41f vertex = {
                        verticies[edges[e][v]][0] * scale_camera,
                        verticies[edges[e][v]][1] * scale_camera,
                        verticies[edges[e][v]][2] * scale_camera,
                        1
                    };
                    cv::Matx44f pose = opengl_coordinate_system(
                        rt_to_transform(
                            frame.rotation, 
                            frame.translation
                        ).inv()
                    );
                    vertex = pose * vertex;
                    glVertex3f(vertex(0), vertex(1), vertex(2));
                }
            }
        }
        glEnd();
        // Render trajectory.
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        for (size_t index = 1; index < mapp.frames.size(); ++index) {
            for (int offset = -1; offset < 1; ++offset) {
                cv::Matx41f vertex = { 0, 0, 0, 1 };
                const Frame& frame = mapp.frames.at(static_cast<int>(index) + offset);
                cv::Matx44f pose = opengl_coordinate_system(
                    rt_to_transform(
                        frame.rotation, 
                        frame.translation
                    ).inv()
                );
                vertex = pose * vertex;
                glVertex3f(vertex(0), vertex(1), vertex(2));
            }
        }
        glEnd();
        // Render landmarks.
        glPointSize(10);
        glBegin(GL_POINTS);
        for (const auto& [landmark_id, landmark] : mapp.landmarks) {
            glColor3d(landmark.colour(0), landmark.colour(1), landmark.colour(2));
            glVertex3d(landmark.location(0), -landmark.location(1), -landmark.location(2));
        }
        glEnd();
        // Swap the buffers of the window.
        glfwSwapBuffers(this->window);
        glfwPollEvents();
        if (glfwWindowShouldClose(this->window)) {
            exit(0);
        }
    }
};

// argc = count of command-line arguments (including program name), argv = array of C-strings.
int main(int argc, char* argv[]) {
    std::printf("Launching slam...\n");
    if (argc < 3) { // Checks that the user provided at least two arguments (video file, focal length). If not print usage hint!
        std::printf("Usage %s [video_file.ext] [focal_length]\n", argv[0]); // 
        exit(1);
    }

    std::printf("Loading video...\n");
    std::printf("  video_file:   %s\n", argv[1]); // Shows which file and focal length you’re about to use.
    std::printf("  focal_length: %s\n", argv[2]);
    cv::VideoCapture capture(argv[1]);
    // This F will populate camera’s intrinsic matrix K.
    double F = std::atof(argv[2]); // Converts the focal-length string to a double (e.g. “800” → 800.0). // atof - ascii to float
    std::printf("Loading camera parameters...\n");
    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    cv::Matx33d K = {
        F, 0, static_cast<double>(width)/2.0, // We assume principal centre lies at centre of an image; so image_width/2, image_height/2
        0, F, static_cast<double>(height)/2.0,
        0, 0, 1
    };
    std::printf("Loading display [%d %d]...\n", width, height);
    Display display(width, height);
    std::printf("Loading slam driver...\n");
    SLAM slam; // instantiate SLAM object with default
    std::printf("Processing frames...\n");
    int i = 0;
    while (capture.isOpened()) {
        std::printf("Starting frame %d/%d\n", ++i, frame_count);
        cv::Mat image;
        if (!capture.read(image)) {
            std::fprintf(stderr, "Failed to read frame.\n");
            break;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::resize(image, image, cv::Size(width, height));
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            slam.process_frame(K, image);
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            std::printf("Processing time: %f seconds\n", frame_duration.count());
        }
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            display.render(slam.mapp);
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            std::printf("Rendering time: %f seconds\n", frame_duration.count());
        }
        if  ((0)) {
            cv::Mat screen_capture = display.capture();
            cv::flip(screen_capture, screen_capture, 0);
            static cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(screen_capture.cols, screen_capture.rows));
            video.write(screen_capture);
        }
        std::printf("\n");
    }
}